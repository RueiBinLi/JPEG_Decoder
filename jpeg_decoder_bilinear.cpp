#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <cstring>
#include <chrono>

const double PI = 3.14159265358979323846;

const uint8_t SOI_MARKER  = 0xD8;
const uint8_t EOI_MARKER  = 0xD9;
const uint8_t APP0_MARKER = 0xE0;
const uint8_t DQT_MARKER  = 0xDB;
const uint8_t DHT_MARKER  = 0xC4;
const uint8_t SOF0_MARKER = 0xC0;
const uint8_t SOS_MARKER  = 0xDA;

// ZigZag Table
const int ZZ[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

// Global Tables for Fast IDCT
float g_idct_table[8][8];

void init_tables() {
    const float c_alpha = 1.0f / sqrt(2.0f);
    for (int u = 0; u < 8; u++) {
        for (int x = 0; x < 8; x++) {
            float cu = (u == 0) ? c_alpha : 1.0f;
            g_idct_table[u][x] = cu * cos((2 * x + 1) * u * PI / 16.0);
        }
    }
}

// --- Data Structures ---

struct HuffmanTable {
    int32_t min_code[17];
    int32_t max_code[17];
    int32_t val_ptr[17];
    std::vector<uint8_t> huff_values;
    
    HuffmanTable() {
        std::fill(min_code, min_code + 17, -1);
        std::fill(max_code, max_code + 17, -1);
        std::fill(val_ptr, val_ptr + 17, -1);
    }
};

struct ComponentInfo {
    uint8_t id;
    uint8_t h_samp; 
    uint8_t v_samp;
    uint8_t quant_table_id;
    uint8_t dc_table_id;
    uint8_t ac_table_id;
};

struct SofInfo {
    uint16_t height;
    uint16_t width;
    uint8_t max_h_samp;
    uint8_t max_v_samp;
    std::vector<ComponentInfo> components;
};

struct Image {
    int width;
    int height;
    struct Pixel { uint8_t r, g, b; };
    std::vector<Pixel> pixels;
    Image(int w, int h) : width(w), height(h), pixels(w * h) {}
};

using Block = std::array<float, 64>;

class BitStream {
public:
    BitStream(std::ifstream& ref) : reader(ref), buf(0), count(0) {}

    inline uint8_t get_bit() {
        if (count == 0) {
            char c;
            reader.read(&c, 1);
            buf = (uint8_t)c;
            if (buf == 0xFF) {
                char check;
                reader.read(&check, 1);
            }
            count = 8;
        }
        uint8_t bit = (buf >> (count - 1)) & 0x01;
        count--;
        return bit;
    }

    inline uint16_t read_bits(int n) {
        uint16_t val = 0;
        for (int i = 0; i < n; i++) {
            val = (val << 1) | get_bit();
        }
        return val;
    }

    uint8_t decode_huffman(const HuffmanTable& table) {
        int code = 0;
        for (int len = 1; len <= 16; len++) {
            code = (code << 1) | get_bit();
            if (code <= table.max_code[len]) {
                int index = table.val_ptr[len] + (code - table.min_code[len]);
                return table.huff_values[index];
            }
        }
        return 0;
    }

private:
    std::ifstream& reader;
    uint8_t buf;
    int count;
};

// --- Main Decoder ---

class JPEGDecoder {
    std::ifstream reader;
    SofInfo sof_info;
    std::array<Block, 4> quant_tables;
    std::array<HuffmanTable, 4> dc_tables;
    std::array<HuffmanTable, 4> ac_tables;
    float last_dc[3] = {0.0f, 0.0f, 0.0f};

public:
    JPEGDecoder(const std::string& filename) {
        reader.open(filename, std::ios::binary);
        if (!reader.is_open()) throw std::runtime_error("File not found");
        for(auto& q : quant_tables) q.fill(0.0f);
    }

    Image decode() {
        uint8_t marker;
        while (read_marker(marker)) {
            switch(marker) {
                case DQT_MARKER: read_dqt(); break;
                case DHT_MARKER: read_dht(); break;
                case SOF0_MARKER: read_sof0(); break;
                case SOS_MARKER: 
                    read_sos();
                    {
                        Image final_img = process_scan();
                        return final_img;
                    }
                case EOI_MARKER: 
                    return Image(0,0);
                case APP0_MARKER: 
                    skip_segment(); 
                    break;
                default:
                    if ((marker >= 0xD0 && marker <= 0xD9) || marker == 0x01) {
                        // Restart markers or reserved - no length to skip
                    } else {
                        skip_segment();
                    }
                    break;
            }
        }
        return Image(0,0);
    }

private:
    uint8_t read_u8() { char c; reader.read(&c, 1); return (uint8_t)c; }
    
    uint16_t read_u16() {
        uint8_t b1 = read_u8();
        uint8_t b2 = read_u8();
        return (b1 << 8) | b2;
    }

    bool read_marker(uint8_t& marker) {
        try {
            uint8_t b = read_u8();
            while (b != 0xFF) {
                b = read_u8();
            }
            
            marker = read_u8();

            while (marker == 0xFF) {
                marker = read_u8();
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    void skip_segment() {
        uint16_t len = read_u16();
        reader.ignore(len - 2);
    }

    void read_dqt() {
        uint16_t len = read_u16();
        len -= 2;
        while (len > 0) {
            uint8_t info = read_u8();
            uint8_t id = info & 0x0F;
            for (int i = 0; i < 64; i++) quant_tables[id][i] = (float)read_u8();
            len -= 65;
        }
    }

    // Build optimized Huffman lookup tables
    void read_dht() {
        uint16_t len = read_u16();
        len -= 2;
        while (len > 0) {
            uint8_t info = read_u8();
            uint8_t type = (info >> 4) & 0x01;
            uint8_t id = info & 0x0F;

            uint8_t bits[17];
            int total = 0;
            for (int i = 1; i <= 16; i++) {
                bits[i] = read_u8();
                total += bits[i];
            }

            HuffmanTable& table = (type == 0) ? dc_tables[id] : ac_tables[id];
            table.huff_values.resize(total);
            for (int i = 0; i < total; i++) table.huff_values[i] = read_u8();

            int32_t code = 0;
            int32_t sym_idx = 0;
            for (int i = 1; i <= 16; i++) {
                if (bits[i] == 0) {
                    table.min_code[i] = -1;
                    table.max_code[i] = -1;
                } else {
                    table.min_code[i] = code;
                    table.max_code[i] = code + bits[i] - 1;
                    table.val_ptr[i] = sym_idx;
                    sym_idx += bits[i];
                    code = (code + bits[i]);
                }
                code <<= 1;
            }
            len -= (1 + 16 + total);
        }
    }

    void read_sof0() {
        read_u16(); 
        read_u8();
        sof_info.height = read_u16();
        sof_info.width = read_u16();
        uint8_t n = read_u8();
        sof_info.components.clear();
        sof_info.max_h_samp = 0;
        sof_info.max_v_samp = 0;
        for (int i = 0; i < n; i++) {
            ComponentInfo c;
            c.id = read_u8();
            uint8_t samp = read_u8();
            c.h_samp = samp >> 4;
            c.v_samp = samp & 0x0F;
            c.quant_table_id = read_u8();
            if (c.h_samp > sof_info.max_h_samp) sof_info.max_h_samp = c.h_samp;
            if (c.v_samp > sof_info.max_v_samp) sof_info.max_v_samp = c.v_samp;
            sof_info.components.push_back(c);
        }
    }

    void read_sos() {
        read_u16();
        uint8_t n = read_u8();
        for (int i = 0; i < n; i++) {
            uint8_t id = read_u8();
            uint8_t t = read_u8();
            for (auto& c : sof_info.components) {
                if (c.id == id) {
                    c.dc_table_id = t >> 4;
                    c.ac_table_id = t & 0x0F;
                    break;
                }
            }
        }
        read_u8(); read_u8(); read_u8();
    }

    // --- Fast Processing ---

    void fast_idct_8x8(float* block) {
        float temp[64];

        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                float sum = 0.0f;
                for (int u = 0; u < 8; u++) {
                    sum += g_idct_table[u][x] * block[y * 8 + u];
                }
                temp[y * 8 + x] = sum;
            }
        }

        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                float sum = 0.0f;
                for (int v = 0; v < 8; v++) {
                     sum += g_idct_table[v][y] * temp[v * 8 + x];
                }
                block[y * 8 + x] = 0.25f * sum;
            }
        }
    }

    void decode_block(BitStream& bs, HuffmanTable& dc, HuffmanTable& ac, int& prev_dc, int quant_id, float* out_block) {
        std::fill(out_block, out_block + 64, 0.0f);

        // DC
        uint8_t len = bs.decode_huffman(dc);
        int32_t val = 0;
        if (len > 0) {
            uint16_t bits = bs.read_bits(len);
            val = (bits < (1 << (len - 1))) ? (bits - (1 << len) + 1) : bits;
        }
        prev_dc += val;
        out_block[0] = (float)prev_dc * quant_tables[quant_id][0];

        // AC
        int k = 1;
        while (k < 64) {
            uint8_t sym = bs.decode_huffman(ac);
            if (sym == 0) break;
            int z = sym >> 4;
            int l = sym & 0x0F;
            if (z == 15 && l == 0) { k += 16; continue; } // ZRL
            k += z;
            if (k >= 64) break;
            
            int32_t ac_val = 0;
            if (l > 0) {
                uint16_t bits = bs.read_bits(l);
                ac_val = (bits < (1 << (l - 1))) ? (bits - (1 << l) + 1) : bits;
            }

            out_block[ZZ[k]] = (float)ac_val * quant_tables[quant_id][k];
            k++;
        }
        
        fast_idct_8x8(out_block);
    }

    inline uint8_t clamp(float v) {
        return (v < 0) ? 0 : ((v > 255) ? 255 : (uint8_t)v);
    }

    inline float get_comp_sample(const std::vector<Block>& blocks, int h_samp, int x, int y) {
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        int logic_w = h_samp * 8;
        int logic_h = (int)blocks.size() / h_samp * 8;
        
        if (x >= logic_w) x = logic_w - 1;
        if (y >= logic_h) y = logic_h - 1;

        int blk_x = x / 8;
        int blk_y = y / 8;
        int off_x = x % 8;
        int off_y = y % 8;
        
        int blk_idx = blk_y * h_samp + blk_x;
        if (blk_idx >= blocks.size()) return 0.0f;
        return blocks[blk_idx][off_y * 8 + off_x];
    }

    void upsample_chroma_block(const float* src, float* dst) {
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                float current = src[y * 8 + x];
                float right   = (x < 7) ? src[y * 8 + (x + 1)] : current;
                float bottom  = (y < 7) ? src[(y + 1) * 8 + x] : current;
                float diag    = (x < 7 && y < 7) ? src[(y + 1) * 8 + (x + 1)] : 
                                ((x < 7) ? right : ((y < 7) ? bottom : current));

                int out_idx = (y * 2) * 16 + (x * 2);
                
                dst[out_idx] = current;
                
                dst[out_idx + 1] = (current + right) * 0.5f;

                dst[out_idx + 16] = (current + bottom) * 0.5f;

                dst[out_idx + 17] = (current + right + bottom + diag) * 0.25f;
            }
        }
    }

    Image process_scan() {
        BitStream bs(reader);
        Image img(sof_info.width, sof_info.height);
        
        int mcu_w = 8 * sof_info.max_h_samp;
        int mcu_h = 8 * sof_info.max_v_samp;
        int num_x = (sof_info.width + mcu_w - 1) / mcu_w;
        int num_y = (sof_info.height + mcu_h - 1) / mcu_h;
        
        std::vector<std::vector<Block>> mcu_data(sof_info.components.size());
        for(size_t i=0; i<sof_info.components.size(); ++i) {
            int num_blocks = sof_info.components[i].h_samp * sof_info.components[i].v_samp;
            mcu_data[i].resize(num_blocks);
        }

        bool is_420 = (sof_info.components.size() == 3) &&
                      (sof_info.components[0].h_samp == 2 && sof_info.components[0].v_samp == 2) &&
                      (sof_info.components[1].h_samp == 1 && sof_info.components[1].v_samp == 1) &&
                      (sof_info.components[2].h_samp == 1 && sof_info.components[2].v_samp == 1);

        std::vector<float> cb_upsampled(16 * 16);
        std::vector<float> cr_upsampled(16 * 16);

        for (int my = 0; my < num_y; my++) {
            for (int mx = 0; mx < num_x; mx++) {
                
                for (int i = 0; i < sof_info.components.size(); i++) {
                    auto& comp = sof_info.components[i];
                    int dc_idx = (comp.id == 1) ? 0 : (comp.id == 2 ? 1 : 2);
                    for (int b = 0; b < mcu_data[i].size(); b++) {
                         decode_block(bs, dc_tables[comp.dc_table_id], ac_tables[comp.ac_table_id], 
                                      reinterpret_cast<int&>(last_dc[dc_idx]), comp.quant_table_id, mcu_data[i][b].data());
                    }
                }

                
                if (is_420) {
                    upsample_chroma_block(mcu_data[1][0].data(), cb_upsampled.data());
                    upsample_chroma_block(mcu_data[2][0].data(), cr_upsampled.data());

                    for (int y = 0; y < 16; y++) {
                        int py = my * 16 + y;
                        if (py >= img.height) break;
                        
                        int y_blk_row = y / 8;
                        int y_pixel_row = y % 8;

                        for (int x = 0; x < 16; x++) {
                            int px = mx * 16 + x;
                            if (px >= img.width) break;

                            int y_blk_col = x / 8;
                            int y_pixel_col = x % 8;
                            int y_blk_idx = y_blk_row * 2 + y_blk_col;
                            
                            float Y = mcu_data[0][y_blk_idx][y_pixel_row * 8 + y_pixel_col];
                            
                            float Cb = cb_upsampled[y * 16 + x];
                            float Cr = cr_upsampled[y * 16 + x];

                            int r = (int)(Y + 1.402f * Cr + 128.0f);
                            int g = (int)(Y - 0.34414f * Cb - 0.71414f * Cr + 128.0f);
                            int b = (int)(Y + 1.772f * Cb + 128.0f);

                            img.pixels[py * img.width + px] = {
                                (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r)),
                                (uint8_t)(g < 0 ? 0 : (g > 255 ? 255 : g)),
                                (uint8_t)(b < 0 ? 0 : (b > 255 ? 255 : b))
                            };
                        }
                    }

                } else {
                    for (int y = 0; y < mcu_h; y++) {
                        for (int x = 0; x < mcu_w; x++) {
                            int px = mx * mcu_w + x;
                            int py = my * mcu_h + y;
                            if (px >= img.width || py >= img.height) continue;

                            float y_val = 0, cb_val = 0, cr_val = 0;
                            
                            auto get_val_nn = [&](int c_idx) {
                                auto& c = sof_info.components[c_idx];
                                int sx = (x * c.h_samp) / sof_info.max_h_samp;
                                int sy = (y * c.v_samp) / sof_info.max_v_samp;
                                int blk_idx = (sy / 8) * c.h_samp + (sx / 8);
                                return mcu_data[c_idx][blk_idx][(sy % 8) * 8 + (sx % 8)];
                            };

                            if (sof_info.components.size() >= 1) y_val = get_val_nn(0);
                            if (sof_info.components.size() >= 3) {
                                cb_val = get_val_nn(1);
                                cr_val = get_val_nn(2);
                            }

                            img.pixels[py * img.width + px] = {
                                clamp(y_val + 1.402f * cr_val + 128.0f),
                                clamp(y_val - 0.34414f * cb_val - 0.71414f * cr_val + 128.0f),
                                clamp(y_val + 1.772f * cb_val + 128.0f)
                            };
                        }
                    }
                }
            }
        }
        return img;
    }
};

void write_ppm(const std::string& filename, const Image& img) {
    std::ofstream out(filename, std::ios::binary);
    out << "P6\n" << img.width << " " << img.height << "\n255\n";
    for (const auto& p : img.pixels) {
        out.write((char*)&p.r, 1);
        out.write((char*)&p.g, 1);
        out.write((char*)&p.b, 1);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./jpeg_opt <input.jpg>\n";
        return 1;
    }
    
    init_tables();

    try {
        JPEGDecoder decoder(argv[1]);

        std::cout << "Start decoding..." << std::endl;

        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();

        Image img = decoder.decode();

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

        write_ppm("out_interpolation.ppm", img);
        
        std::cout << "Decoded to out_interpolation.ppm\n";
        std::cout << "-----------------------------------\n";
        std::cout << "Decoding Time: " << elapsed.count() << " ms\n";
        std::cout << "-----------------------------------\n";

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}