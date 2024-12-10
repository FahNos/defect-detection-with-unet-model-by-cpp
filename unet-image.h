#pragma once

#include <string>
#include <vector>
#include <cassert>

struct unet_image {
    int w, h, c;
    std::vector<float> data;

    unet_image() : w(0), h(0), c(0) {}
    unet_image(int w, int h, int c) : w(w), h(h), c(c), data(w*h*c) {}

    float get_pixel(int x, int y, int c) const {
        assert(x >= 0 && x < w && y >= 0 && y < h && c >= 0 && c < this->c);
        return data[c*w*h + y*w + x];
    }

    void set_pixel(int x, int y, int c, float val) {
        assert(x >= 0 && x < w && y >= 0 && y < h && c >= 0 && c < this->c);
        data[c*w*h + y*w + x] = val;
    }

    void add_pixel(int x, int y, int c, float val) {
        assert(x >= 0 && x < w && y >= 0 && y < h && c >= 0 && c < this->c);
        data[c*w*h + y*w + x] += val;
    }

    void fill(float val) {
        std::fill(data.begin(), data.end(), val);
    }
};

bool load_unet_image(const char *fname, unet_image & img);
unet_image letterbox_image_unet(const unet_image & im, int w, int h);
bool save_unet_image(const unet_image & im, const char *name, int quality);

