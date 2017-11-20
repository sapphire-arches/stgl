#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdint.h>

#include "arg.h"

#define Glyph Glyph_
#define Font Font_

#include "st.h"
#include "rendering.h"

#include <cstddef>
#include <glad/glad.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include FT_BITMAP_H

#define ATLAS_SIZE 4096

#define POSITION_LOCATION 0
#define COLOR_LOCATION 1
#define UV_LOCATION 2

#define TEXTURE_BINDING 0

#define TRANSFORM_LOCATION 0
#define TEXTURE_LOCATION 1

static const char * vert_shader =
  "#version 450\n"
  "layout(location=0) in vec3 position;\n"
  "layout(location=1) in vec4 in_color;\n"
  "layout(location=2) in vec2 in_uv;\n"
  "layout(location=0) uniform mat4 transform;\n"
  "out vec2 cross_uv;\n"
  "out vec4 cross_color;\n"
  "void main() {"
  "  gl_Position = transform * vec4(position, 1);\n"
  "  cross_uv = in_uv;\n"
  "  cross_color = in_color;\n"
  "}\n";

static const char * frag_shader =
  "#version 450\n"
  "out vec4 color;\n"
  "in vec4 cross_color;\n"
  "in vec2 cross_uv;\n"
  "layout(location=1, binding=0) uniform sampler2D font_tex;\n"
  "void main() {\n"
  "  color = cross_color * texture(font_tex, cross_uv).r;\n"
  //   "  color = cross_color; \n"
  "}\n";

struct __attribute__((packed)) vec2 {
  float x, y;
};

struct rect {
  vec2 origin;
  vec2 size;
};

struct __attribute__((packed)) vertex {
  struct __attribute__((packed)) {
    float x, y, z;
  } pos;
  struct vec2 texcoords;
  struct color c;
};

class GlTexture;
template<typename VertexClass> class GlBuffer;
template<typename VertexClass> class GlVAO;

class GlTexture {
private:
  GLuint _id;
public:
  GlTexture() {
    glGenTextures(1, &_id);
  }

  GlTexture(GlTexture & other) = delete;

  GlTexture(GlTexture && other) {
    _id = other._id;
  }

  GlTexture & operator=(GlTexture & other) = delete;

  GlTexture & operator=(GlTexture && other) {
    _id = other._id;
    return *this;
  }

  ~GlTexture() {
    glDeleteTextures(1, &_id);
  }

  GLuint GetID() {
    return _id;
  }
};

template<typename T>
class GlBuffer {
private:
  /** Name of the buffer under management */
  GLuint _id;
  /** The number of bytes allocated in that buffer */
  size_t _capacity;
  /** CPU-side storage for all the information */
  std::vector<T> _storage;

  friend GlVAO<T>;
public:
  GlBuffer() {
    glCreateBuffers(1, &_id);
    _storage.reserve(1); // Make sure there is room for at least 1 element
    _capacity = _storage.capacity() * sizeof(T);
    glNamedBufferData(_id, _capacity, NULL, GL_STREAM_DRAW);
  }

  GlBuffer(GlBuffer & other) = delete;

  GlBuffer(GlBuffer && other) {
    _id = other._id;
    _capacity = other._capacity;
    _storage = std::move(other._storage);
  }

  GlBuffer & operator= (GlBuffer & other) = delete;

  GlBuffer & operator= (GlBuffer && other) {
    _id = other._id;
    _capacity = other._capacity;
    _storage = std::move(other._storage);
    return *this;
  }

  void push_elements(T * elems, size_t count) {
    if (_storage.size() + count > 65535) {
      fprintf(stderr, "Warning: too many elements, not adding to buffer\n");
      return;
    }
    _storage.insert(_storage.end(), elems, elems + count);
  }

  void sync() {
    // printf("Syncing %d objects\n", _storage.size());
    size_t bsize = _storage.size() * sizeof(T);
    if (_capacity < bsize) {
      _capacity = bsize;
      glNamedBufferData(_id, _capacity, NULL, GL_STREAM_DRAW);
    }
    glNamedBufferSubData(_id, 0, bsize, _storage.data());
  }

  void clear() {
    _storage.clear();
  }

  typename std::vector<T>::size_type num_elems() const {
    return _storage.size();
  }
};

class GlShader {
private:
  GLuint _prog_id;

  GLuint AllocShader(GLenum shader_type, std::string & src);
public:
  /** Creates a new shader

      @param vert the vertex shader
      @param frag the fragment shader
  */
  GlShader(std::string vert, std::string frag);
  GlShader(GlShader & other) = delete;
  GlShader(GlShader && other);
  GlShader & operator=(GlShader & other) = delete;
  GlShader & operator=(GlShader && other);
  ~GlShader();

  void bind(void);
  void uniform(GLint location, GLboolean transpose, float matrix[16]);
};

static void PrintShaderInfoLog(GLuint shader) {
  int len;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
  if (len > 0) {
    char * buf = (char*)malloc(len + 1);
    glGetShaderInfoLog(shader, len + 1, &len, buf);
    printf("%s\n", buf);
    free(buf);
  }
}

GLuint GlShader::AllocShader(GLenum shader_type, std::string & src) {
  GLuint shdr = glCreateShader(shader_type);
  int len = src.length();
  const char * const csrc = src.c_str();
  glShaderSource(shdr, 1, &csrc, &len);
  glCompileShader(shdr);
  PrintShaderInfoLog(shdr);
  int compiled;
  glGetShaderiv(shdr, GL_COMPILE_STATUS, &compiled);
  if (compiled != GL_TRUE) {
    die("unable to compile shader\n");
  }

  return shdr;
}

GlShader::GlShader(std::string vert, std::string frag) {
  printf("Make vertex shader\n");
  GLuint vshdr = AllocShader(GL_VERTEX_SHADER, vert);
  printf("Make fragment shader\n");
  GLuint fshdr = AllocShader(GL_FRAGMENT_SHADER, frag);

  _prog_id = glCreateProgram();
  glAttachShader(_prog_id, vshdr);
  glAttachShader(_prog_id, fshdr);
  glLinkProgram(_prog_id);

  glDeleteShader(vshdr);
  glDeleteShader(fshdr);

  int linked;
  glGetProgramiv(_prog_id, GL_LINK_STATUS, &linked);
  if (linked != GL_TRUE) {
    int size;
    glGetProgramiv(_prog_id, GL_INFO_LOG_LENGTH, &size);
    void * data = malloc(size + 1);
    glGetProgramInfoLog(_prog_id, size, &size, (char*)data);
    printf("%s\n", data);
    free(data);
    die("Unable to link program\n");
  }
}

GlShader::~GlShader() {
  glDeleteProgram(_prog_id);
}

void GlShader::bind() {
  glUseProgram(_prog_id);
}

void GlShader::uniform(GLint location, GLboolean transpose, float matrix[16]) {
  glProgramUniformMatrix4fv(_prog_id, location, 1, transpose, matrix);
}

template<typename VertexType>
class GlVAO {
public:
  typedef std::shared_ptr<GlBuffer<VertexType>> buffer_ref;
  typedef std::shared_ptr<const GlBuffer<VertexType>> buffer_const_ref;
private:
  buffer_ref _buffer;
  GLuint _id;
public:
  GlVAO(buffer_ref buffer);
  GlVAO(const GlVAO & other) = delete;
  GlVAO(GlVAO && other);
  GlVAO & operator= (const GlVAO & other) = delete;
  GlVAO & operator= (GlVAO && other);
  ~GlVAO();

  void bind_buffer(buffer_const_ref buffer, GLuint buffer_index, GLintptr offset, GLsizei stride);
  void bind_attrib(GLuint attrib_index, GLuint buffer_index);
  void enable_attrib(GLuint attrib_index);
  void attrib_format(GLuint attrib_index, GLint size, GLenum type, GLboolean normalized, GLuint relativeOffset);
  void bind() const;
};

template<typename VertexType>
GlVAO<VertexType>::GlVAO(std::shared_ptr<GlBuffer<VertexType>> buffer):
  _buffer(buffer) {
  glCreateVertexArrays(1, &_id);
}

template<typename V>
GlVAO<V>::~GlVAO() {
  glDeleteVertexArrays(1, &_id);
}

template<typename V>
void GlVAO<V>::bind_buffer(buffer_const_ref buffer, GLuint buffer_index, GLintptr offset, GLsizei stride) {
  glVertexArrayVertexBuffer(_id, buffer_index, buffer->_id, offset, stride);
}

template<typename V>
void GlVAO<V>::bind_attrib(GLuint attrib_index, GLuint buffer_index) {
  glVertexArrayAttribBinding(_id, attrib_index, buffer_index);
}

template<typename V>
void GlVAO<V>::attrib_format(GLuint attrib_index, GLint size, GLenum type, GLboolean normalized, GLuint relative_offset) {
  glVertexArrayAttribFormat(_id, attrib_index, size, type, normalized, relative_offset);
}

template<typename V>
void GlVAO<V>::enable_attrib(GLuint attrib_index) {
  glEnableVertexArrayAttrib(_id, attrib_index);
}


template<typename V>
void GlVAO<V>::bind() const {
  glBindVertexArray(_id);
}

struct glyph_render_params{
  struct rect uvs;
  struct vec2 offset;
};

struct atlas {
  GlTexture tex;
  FT_Face face;
  std::unordered_map<FT_UInt, struct glyph_render_params> uvs;
  unsigned int cx, cy, rowmax;

  atlas(FT_Face my_face):
    tex(),
    face(my_face) {
    glBindTexture(GL_TEXTURE_2D, tex.GetID());
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8, ATLAS_SIZE, ATLAS_SIZE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    cx = 0;
    cy = 0;
    rowmax = 0;
  }

  void glyph_render_params(FT_UInt glyph, struct glyph_render_params & out) {
    auto kv = uvs.find(glyph);
    if (kv != uvs.end()) {
      out = kv->second;
      return;
    }
    // render the glyph
    int error;
    if ((error = FT_Load_Glyph(face, glyph, FT_LOAD_DEFAULT)) != 0) {
      die("Failed to load glyph %d %d\n", glyph, error);
    }
    if ((error = FT_Render_Glyph(face->glyph, FT_RENDER_MODE_NORMAL)) != 0) {
      die("Failed to render glyph %d %d\n", glyph, error);
    }
    FT_Bitmap * bm = &face->glyph->bitmap;
    if (cx + bm->width >= ATLAS_SIZE) {
      cx = 0;
      cy += rowmax + 1;
      if (cy >= ATLAS_SIZE) {
        die("Ran out of space in atlas!\n");
      }
      rowmax = 0;
    }
    if (bm->rows > rowmax) {
      rowmax = bm->rows;
    }
    out.offset.x = face->glyph->bitmap_left;
    out.offset.y = face->glyph->bitmap_top;
    out.uvs.origin.x = cx / (float)ATLAS_SIZE;
    out.uvs.origin.y = cy / (float)ATLAS_SIZE;
    out.uvs.size.x = bm->width / (float)ATLAS_SIZE;
    out.uvs.size.y = bm->rows / (float)ATLAS_SIZE;

    glBindTexture(GL_TEXTURE_2D, tex.GetID());
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, cx, cy, bm->width, bm->rows,
                    GL_RED, GL_UNSIGNED_BYTE, bm->buffer);

    cx += bm->width + 1;

    uvs.insert({glyph, out});
  }
};

template<typename VertexType>
class RenderJob {
private:
  std::shared_ptr<GlTexture> texture;
  std::shared_ptr<GlBuffer<VertexType>> verts;
};

struct render_context {
  GlShader _shader;
  std::vector<RenderJob<vertex>> _jobs;
  std::shared_ptr<GlBuffer<vertex>> _verts;
  std::shared_ptr<GlVAO<vertex>> _vert_vao;
  int _win_w;
  int _win_h;

  void set_size(int w, int h);
  void do_render();
  void render_rune(const glyph_spec * spec);

  render_context();
};

void render_context::do_render() {
  _verts->sync();
  _shader.bind();

  // Transform matrix in row major order
  float transform[16] = {
    2.f / _win_w, 0.f, 0.f, -1.f,
    0.f, -2.f / _win_h, 0.f, 1.f,
    0.f, 0.f, 1.f, 0.f,
    0.f, 0.f, 0.f, 1.f,
  };

  _shader.uniform(TRANSFORM_LOCATION, GL_TRUE, transform);

  _vert_vao->bind();
  glDrawArrays(GL_TRIANGLES, 0, _verts->num_elems());

  _verts->clear();
}

void render_context::render_rune(const glyph_spec * spec) {
  struct glyph_render_params rps;
  spec->font->glyph_render_params(spec->glyph, rps);

  float pixel_width = ATLAS_SIZE * rps.uvs.size.x;
  float pixel_height = ATLAS_SIZE * rps.uvs.size.y;
  float base_x = rps.offset.x + static_cast<float>(spec->x);
  float base_y = -rps.offset.y + static_cast<float>(spec->y);

  vertex lverts[4] = {
    {
      .pos = {
        .x = base_x,
        .y = base_y,
        .z = 0,
      },
      .texcoords = rps.uvs.origin,
      .c = *spec->c
    },
    {
      .pos = {
        .x = base_x + pixel_width,
        .y = base_y + pixel_height,
        .z = 0,
      },
      .texcoords = {
        .x = rps.uvs.origin.x + rps.uvs.size.x,
        .y = rps.uvs.origin.y + rps.uvs.size.y,
      },
      .c = *spec->c
    },
    {
      .pos = {
        .x = base_x,
        .y = base_y + pixel_height,
        .z = 0,
      },
      .texcoords = {
        .x = rps.uvs.origin.x,
        .y = rps.uvs.origin.y + rps.uvs.size.y,
      },
      .c = *spec->c
    },
    {
      .pos = {
        .x = base_x + pixel_width,
        .y = base_y,
        .z = 0,
      },
      .texcoords = {
        .x = rps.uvs.origin.x + rps.uvs.size.x,
        .y = rps.uvs.origin.y,
      },
      .c = *spec->c
    },
  };

  _verts->push_elements(lverts, 3);
  lverts[2] = lverts[3];
  _verts->push_elements(lverts, 3);
}

void render_context::set_size(int w, int h) {
  _win_w = w;
  _win_h = h;
}

render_context::render_context() :
  _shader(std::string(vert_shader), std::string(frag_shader)),
  _verts(new GlBuffer<vertex>()),
  _vert_vao(new GlVAO<vertex>(_verts)),
  _win_w(1),
  _win_h(1)
{
  _vert_vao->bind_buffer(_verts, 0, 0, sizeof(vertex));

  _vert_vao->enable_attrib(POSITION_LOCATION);
  _vert_vao->enable_attrib(UV_LOCATION);
  _vert_vao->enable_attrib(COLOR_LOCATION);

  _vert_vao->bind_attrib(POSITION_LOCATION, 0);
  _vert_vao->bind_attrib(UV_LOCATION, 0);
  _vert_vao->bind_attrib(COLOR_LOCATION, 0);

  _vert_vao->attrib_format(POSITION_LOCATION, 3, GL_FLOAT, GL_FALSE, offsetof(vertex, pos));
  _vert_vao->attrib_format(UV_LOCATION, 2, GL_FLOAT, GL_FALSE, offsetof(vertex, texcoords));
  _vert_vao->attrib_format(COLOR_LOCATION, 4, GL_FLOAT, GL_FALSE, offsetof(vertex, c));
}


////////////////////////////////////////////////////////////////////////////////
// External function implementation
////////////////////////////////////////////////////////////////////////////////
struct render_context * render_init() {
  return new render_context();
}

void render_destroy(struct render_context * rc) {
  delete rc;
}

void render_resize(struct render_context * rc, int w, int h) {
  rc->set_size(w, h);
}

void render_do_render(struct render_context * rc) {
  rc->do_render();
}

struct atlas * atlas_create_from_face(FT_Face f) {
  return new atlas(f);
}

struct atlas * atlas_create_from_pattern(FT_Library lib, FcPattern * pat) {
  char * file_name;
  int file_index;
  if (FcPatternGetString(pat, FC_FILE, 0, (FcChar8**)&file_name) != FcResultMatch) {
    fputs("st: failed to get font file\n", stderr);
    return nullptr;
  }
  if (FcPatternGetInteger(pat, FC_INDEX, 0, &file_index) != FcResultMatch) {
    fputs("st: failed to get font index\n", stderr);
    return nullptr;
  }
  FT_Face f;
  if (!FT_New_Face(lib, file_name, file_index, &f)) {
    fputs("st: failed to open font file\n", stderr);
    return nullptr;
  }
  return atlas_create_from_face(f);
}

void atlas_destroy(struct atlas * a, bool del_face) {
  if (del_face) {
    FT_Done_Face(a->face);
  }
  delete a;
}

FT_Face atlas_get_face(struct atlas * a) {
  return a->face;
}

void render_rune(struct render_context * rc, const struct glyph_spec * spec) {
  rc->render_rune(spec);
}
