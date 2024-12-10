#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "unet-image.h"

bool save_unet_image(const unet_image & im, const char *name, int quality)
{
    uint8_t *data = (uint8_t*)calloc(im.w*im.h*im.c, sizeof(uint8_t));
    for (int k = 0; k < im.c; ++k) {
        for (int i = 0; i < im.w*im.h; ++i) {
            data[i*im.c+k] = (uint8_t) (im.data[i + k*im.w*im.h]);
        }
    }
    int success = stbi_write_jpg(name, im.w, im.h, im.c, data, quality);
    free(data);
    if (!success) {
        fprintf(stderr, "Failed to write image %s\n", name);
        return false;
    }
    return true;
}

bool load_unet_image(const char *fname, unet_image & img)
{
    int w, h, c;
    uint8_t * data = stbi_load(fname, &w, &h, &c, 3);
    if (!data) {
        return false;
    }
    c = 3;
    img.w = w;
    img.h = h;
    img.c = c;
    img.data.resize(w*h*c);
    for (int k = 0; k < c; ++k){
        for (int j = 0; j < h; ++j){
            for (int i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                img.data[dst_index] = (float)data[src_index];
            }
        }
    }
    stbi_image_free(data);
    return true;
}

static unet_image resize_image(const unet_image & im, int w, int h)
{
    unet_image resized(w, h, im.c);
    unet_image part(w, im.h, im.c);
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for (int k = 0; k < im.c; ++k){
        for (int r = 0; r < im.h; ++r) {
            for (int c = 0; c < w; ++c) {
                float val = 0;
                if (c == w-1 || im.w == 1){
                    val = im.get_pixel(im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * im.get_pixel(ix, r, k) + dx * im.get_pixel(ix+1, r, k);
                }
                part.set_pixel(c, r, k, val);
            }
        }
    }
    for (int k = 0; k < im.c; ++k){
        for (int r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for (int c = 0; c < w; ++c){
                float val = (1-dy) * part.get_pixel(c, iy, k);
                resized.set_pixel(c, r, k, val);
            }
            if (r == h-1 || im.h == 1) continue;
            for (int c = 0; c < w; ++c){
                float val = dy * part.get_pixel(c, iy+1, k);
                resized.add_pixel(c, r, k, val);
            }
        }
    }
    return resized;
}

static void embed_image(const unet_image & source, unet_image & dest, int dx, int dy)
{
    for (int k = 0; k < source.c; ++k) {
        for (int y = 0; y < source.h; ++y) {
            for (int x = 0; x < source.w; ++x) {
                float val = source.get_pixel(x, y, k);
                dest.set_pixel(dx+x, dy+y, k, val);
            }
        }
    }
}

unet_image letterbox_image_unet(const unet_image & im, int w, int h)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w/im.w) < ((float)h/im.h)) {
        new_w = w;
        new_h = (im.h * w)/im.w;
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
    unet_image resized = resize_image(im, new_w, new_h);
    unet_image boxed(w, h, im.c);
    boxed.fill(0.5);
    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2);
    return boxed;
}