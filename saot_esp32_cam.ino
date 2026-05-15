#include "esp_camera.h"
#include <Arduino.h>

#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

struct ColorRange {
  uint8_t r_min, r_max;
  uint8_t g_min, g_max;
  uint8_t b_min, b_max;
};

ColorRange tm_range = {150, 255, 80, 160, 0, 80};
ColorRange def_range = {150, 255, 0, 80, 0, 80};

void setup() {
  Serial.begin(115200);
  Serial.println("[SAOT] Initializing ESP32-CAM...");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_CIF;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] Camera init failed with error 0x%x", err);
    return;
  }

  Serial.println("[OK] Camera ready.");
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[ERROR] Frame capture failed");
    return;
  }

  long tm_x_sum = 0, tm_y_sum = 0, tm_count = 0;
  long def_x_sum = 0, def_y_sum = 0, def_count = 0;

  int w = fb->width;
  int h = fb->height;
  uint16_t *buf = (uint16_t *)fb->buf;

  for (int y = 0; y < h; y += 2) {
    for (int x = 0; x < w; x += 2) {
      uint16_t pixel = buf[y * w + x];

      uint8_t r = (pixel >> 8) & 0xF8;
      uint8_t g = (pixel >> 3) & 0xFC;
      uint8_t b = (pixel << 3) & 0xF8;

      if (r >= tm_range.r_min && r <= tm_range.r_max && g >= tm_range.g_min &&
          g <= tm_range.g_max && b >= tm_range.b_min && b <= tm_range.b_min) {
        tm_x_sum += x;
        tm_y_sum += y;
        tm_count++;
      }

      if (r >= def_range.r_min && r <= def_range.r_max &&
          g >= def_range.g_min && g <= def_range.g_max &&
          b >= def_range.b_min && b <= def_range.b_min) {
        def_x_sum += x;
        def_y_sum += y;
        def_count++;
      }
    }
  }

  if (tm_count > 50 || def_count > 50) {
    float tm_x = (tm_count > 0) ? (float)tm_x_sum / tm_count : -1;
    float tm_y = (tm_count > 0) ? (float)tm_y_sum / tm_count : -1;
    float def_x = (def_count > 0) ? (float)def_x_sum / def_count : -1;
    float def_y = (def_count > 0) ? (float)def_y_sum / def_count : -1;

    Serial.printf("DATA:%.2f,%.2f,%.2f,%.2f\n", tm_x, tm_y, def_x, def_y);
  }

  esp_camera_fb_return(fb);
  delay(100);
}
