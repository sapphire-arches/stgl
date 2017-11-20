#ifndef _RENDERING_H_
#define _RENDERING_H_

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <fontconfig/fontconfig.h>
#include <ft2build.h>

#include FT_FREETYPE_H

#ifdef __cplusplus
extern "C" {
#endif

  struct __attribute__((packed)) color {
    float r, g, b, a;
  };

  static int colors_equal(struct color * a, struct color * b) {
    return a->r == b->r && \
      a->g == b->g &&      \
      a->b == b->b &&      \
      a->a == b->a;
  }

  struct glyph_spec {
    FT_UInt glyph;
    struct color * c;
    struct atlas * font;
    int x, y;
  };

  /** Keeps track of the things that need to be rendered */
  struct render_context;
  /** Font atlas. One atlas per font */
  struct atlas;

  struct render_context * render_init(void);
  void render_destroy(struct render_context * rc);
  void render_do_render(struct render_context * rc);
  void render_resize(struct render_context * rc, int w, int h);

  /** Create an atlas. Takes ownership of the font */
  struct atlas * atlas_create_from_face(FT_Face f);
  struct atlas * atlas_create_from_pattern(FT_Library lib, FcPattern * pat);
  /** Destroy an atlas. Second parameter is true if we should also destroy the face */
  void atlas_destroy(struct atlas * a, bool);
  FT_Face atlas_get_face(struct atlas * a);

  void render_rune(struct render_context * rc, const struct glyph_spec * spec);
  void render_rect(struct render_context * rc, const struct color * const c, int x, int y, int w, int h);
#ifdef __cplusplus
}
#endif

#endif
