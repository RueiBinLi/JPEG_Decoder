/**
 * Final Optimized JPEG Decoder
 * Base: The "Interpolation" version (Fastest & Correct Quality)
 * Optimization 1: Lookup Table for Clamping (Branchless)
 * Optimization 2: Integer YCbCr -> RGB Conversion
 * Optimization 3: Pointer arithmetic instead of vector indexing
 */

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

float g_idct_table[8][8];

uint8_t g_clamp_table[1280]; 
uint8_t* g_clamp = &g_clamp_table[512];

void init_tables() {
    const float c_alpha = 1.0f / sqrt(2.0f);
    for (int u = 0; u < 8; u++) {
        for (int x = 0; x < 8; x++) {
            float cu = (u == 0) ? c_alpha : 1.0f;
            g_idct_table[u][x] = cu * cos((2 * x + 1) * u * PI / 16.0);
        }
    }

    for (int i = 0; i < 1280; i++) {
        int val = i - 512;
        if (val < 0) g_clamp_table[i] = 0;
        else if (val > 255) g_clamp_table[i] = 255;
        else g_clamp_table[i] = (uint8_t)val;
    }
}

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
    std::vector<uint8_t> data;
    Image(int w, int h) : width(w), height(h), data(w * h * 3) {}
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
        for (int i = 0; i < n; i++) val = (val << 1) | get_bit();
        return val;
    }

    inline uint8_t decode_huffman(const HuffmanTable& table) {
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
                    return process_scan();
                case EOI_MARKER: return Image(0,0);
                case APP0_MARKER: skip_segment(); break;
                default:
                    if ((marker >= 0xD0 && marker <= 0xD9) || marker == 0x01) {} 
                    else skip_segment();
                    break;
            }
        }
        return Image(0,0);
    }

private:
    uint8_t read_u8() { char c; reader.read(&c, 1); return (uint8_t)c; }
    uint16_t read_u16() { uint8_t b1 = read_u8(); uint8_t b2 = read_u8(); return (b1 << 8) | b2; }

    bool read_marker(uint8_t& marker) {
        try {
            uint8_t b = read_u8();
            while (b != 0xFF) b = read_u8();
            marker = read_u8();
            while (marker == 0xFF) marker = read_u8();
            return true;
        } catch (...) { return false; }
    }
    void skip_segment() { uint16_t len = read_u16(); reader.ignore(len - 2); }

    void read_dqt() {
        uint16_t len = read_u16(); len -= 2;
        while (len > 0) {
            uint8_t info = read_u8();
            uint8_t id = info & 0x0F;
            for (int i = 0; i < 64; i++) quant_tables[id][i] = (float)read_u8();
            len -= 65;
        }
    }

    void read_dht() {
        uint16_t len = read_u16(); len -= 2;
        while (len > 0) {
            uint8_t info = read_u8();
            uint8_t type = (info >> 4) & 0x01;
            uint8_t id = info & 0x0F;
            uint8_t bits[17];
            int total = 0;
            for (int i = 1; i <= 16; i++) { bits[i] = read_u8(); total += bits[i]; }
            HuffmanTable& table = (type == 0) ? dc_tables[id] : ac_tables[id];
            table.huff_values.resize(total);
            for (int i = 0; i < total; i++) table.huff_values[i] = read_u8();
            int32_t code = 0, sym_idx = 0;
            for (int i = 1; i <= 16; i++) {
                if (bits[i] == 0) { table.min_code[i] = -1; table.max_code[i] = -1; }
                else {
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
        read_u16(); read_u8();
        sof_info.height = read_u16();
        sof_info.width = read_u16();
        uint8_t n = read_u8();
        sof_info.components.clear();
        sof_info.max_h_samp = 0; sof_info.max_v_samp = 0;
        for (int i = 0; i < n; i++) {
            ComponentInfo c;
            c.id = read_u8();
            uint8_t samp = read_u8();
            c.h_samp = samp >> 4; c.v_samp = samp & 0x0F;
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
                if (c.id == id) { c.dc_table_id = t >> 4; c.ac_table_id = t & 0x0F; break; }
            }
        }
        read_u8(); read_u8(); read_u8();
    }

    void fast_idct_8x8(float* block) {
        float temp[64];
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                float sum = 0.0f;
                for (int u = 0; u < 8; u++) sum += g_idct_table[u][x] * block[y * 8 + u];
                temp[y * 8 + x] = sum;
            }
        }
        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                float sum = 0.0f;
                for (int v = 0; v < 8; v++) sum += g_idct_table[v][y] * temp[v * 8 + x];
                block[y * 8 + x] = 0.25f * sum;
            }
        }
    }

    void decode_block(BitStream& bs, HuffmanTable& dc, HuffmanTable& ac, int& prev_dc, int quant_id, float* out_block) {
        std::fill(out_block, out_block + 64, 0.0f);
        uint8_t len = bs.decode_huffman(dc);
        int32_t val = 0;
        if (len > 0) {
            uint16_t bits = bs.read_bits(len);
            val = (bits < (1 << (len - 1))) ? (bits - (1 << len) + 1) : bits;
        }
        prev_dc += val;
        out_block[0] = (float)prev_dc * quant_tables[quant_id][0];

        int k = 1;
        while (k < 64) {
            uint8_t sym = bs.decode_huffman(ac);
            if (sym == 0) break;
            int z = sym >> 4;
            int l = sym & 0x0F;
            if (z == 15 && l == 0) { k += 16; continue; }
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
            mcu_data[i].resize(sof_info.components[i].h_samp * sof_info.components[i].v_samp);
        }

        bool is_420 = (sof_info.components.size() == 3) &&
                      (sof_info.components[0].h_samp == 2 && sof_info.components[0].v_samp == 2);

        std::vector<float> cb_upsampled(256);
        std::vector<float> cr_upsampled(256);

        uint8_t* pixel_ptr = img.data.data();
        int img_w = img.width;

        for (int my = 0; my < num_y; my++) {
            for (int mx = 0; mx < num_x; mx++) {
                
                // Decode
                for (int i = 0; i < sof_info.components.size(); i++) {
                    auto& comp = sof_info.components[i];
                    int dc_idx = (comp.id == 1) ? 0 : (comp.id == 2 ? 1 : 2);
                    for (auto& block : mcu_data[i]) {
                         decode_block(bs, dc_tables[comp.dc_table_id], ac_tables[comp.ac_table_id], 
                                      reinterpret_cast<int&>(last_dc[dc_idx]), comp.quant_table_id, block.data());
                    }
                }

                // Render
                if (is_420) {
                    upsample_chroma_block(mcu_data[1][0].data(), cb_upsampled.data());
                    upsample_chroma_block(mcu_data[2][0].data(), cr_upsampled.data());

                    int max_y = std::min(16, img.height - my * 16);
                    int max_x = std::min(16, img.width - mx * 16);

                    for (int y = 0; y < max_y; y++) {
                        int row_start = (my * 16 + y) * img_w * 3 + (mx * 16) * 3;
                        int y_blk_row = y / 8;
                        int y_pixel_row = y % 8;

                        for (int x = 0; x < max_x; x++) {
                            int y_blk_col = x / 8;
                            int y_pixel_col = x % 8;
                            
                            float Y = mcu_data[0][y_blk_row * 2 + y_blk_col][y_pixel_row * 8 + y_pixel_col];
                            float Cb = cb_upsampled[y * 16 + x];
                            float Cr = cr_upsampled[y * 16 + x];

                            int y_int = (int)Y;
                            int cb_int = (int)Cb;
                            int cr_int = (int)Cr;

                            int r = y_int + ((1436 * cr_int) >> 10);
                            int g = y_int - ((352 * cb_int + 731 * cr_int) >> 10);
                            int b = y_int + ((1815 * cb_int) >> 10);
                            
                            // Use Lookup Table for Clamping
                            int p_idx = row_start + x * 3;
                            pixel_ptr[p_idx]     = g_clamp[r + 128];
                            pixel_ptr[p_idx + 1] = g_clamp[g + 128];
                            pixel_ptr[p_idx + 2] = g_clamp[b + 128];
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
    out.write((char*)img.data.data(), img.data.size());
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    init_tables();
    try {
        JPEGDecoder decoder(argv[1]);
        auto start = std::chrono::high_resolution_clock::now();
        Image img = decoder.decode();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        write_ppm("out_optimized.ppm", img); 
        
        std::cout << "Decoded to out_optimized.ppm\n";
        std::cout << "Decoding Time: " << elapsed.count() << " ms\n";
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}