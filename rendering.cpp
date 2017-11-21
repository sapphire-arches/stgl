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
#include <cassert>
#include <glad/glad.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>
#include <cmath>

#include FT_BITMAP_H

#define ATLAS_SIZE 4096

#define PARTICLE_FB_SCALE 3

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
  // "  color = vec4(1.f, 1.f, 1.f, texture(font_tex, cross_uv).r);\n"
  // "  color = vec4(cross_color.rgb, texture(font_tex, cross_uv).r);\n"
  // "  color = cross_color * texture(font_tex, cross_uv).r;\n"
  "  color = cross_color;\n"
  "  color.a *= texture(font_tex, cross_uv).r;\n"
  "}\n";

static const char * framebuffer_frag_shader =
  "#version 450\n"
  "out vec4 color;\n"
  "in vec4 cross_color;\n"
  "in vec2 cross_uv;\n"
  "layout(location=1, binding=0) uniform sampler2D font_tex;\n"
  "void main() {\n"
  "  color = cross_color * texture(font_tex, cross_uv);\n"
  "}\n";

static const char * particle_blit_shader =
  "#version 450\n"
  "out vec4 color;\n"
  "in vec4 cross_color;\n"
  "in vec2 cross_uv;\n"
  "layout(location=1, binding=0) uniform sampler2D font_tex;\n"
  "void main() {\n"
  "  color = cross_color * texture(font_tex, cross_uv);\n"
  "  color.a = texture(font_tex, cross_uv).a;\n"
  "}\n";

static const char * color_shader =
  "#version 450\n"
  "out vec4 color;\n"
  "in vec4 cross_color;\n"
  "in vec2 cross_uv;\n"
  "layout(location=1, binding=0) uniform sampler2D font_tex;\n"
  "void main() {\n"
  "  color = cross_color;\n"
  "  color.a = 1.f;\n"
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
template<typename Updatefunc> class ParticleSystem;

////////////////////////////////////////////////////////////////////////////////
// GlTexture
////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
// GlFrameBuffer
////////////////////////////////////////////////////////////////////////////////

class GlFrameBuffer {
private:
  GLuint _id;
  GLsizei _width, _height;
  GLuint _color_id;
  GLuint _depth_id;
public:
  GlFrameBuffer(GLsizei width, GLsizei height, bool linear, GLenum depth_format);
  GlFrameBuffer(const GlFrameBuffer & other) = delete;
  GlFrameBuffer(GlFrameBuffer && other);
  GlFrameBuffer & operator=(GlFrameBuffer & other) = delete;
  GlFrameBuffer & operator=(GlFrameBuffer && other);
  ~GlFrameBuffer();

  void blit_entirely(GLuint dst) const;
  void blit_entirely(const GlFrameBuffer & dst) const;
  void bind(GLenum target) const;

  GLuint get_main_color() const { return _color_id; }
};

GlFrameBuffer::GlFrameBuffer(GLsizei width, GLsizei height, bool linear, GLenum depth_format)
  : _width(width),
    _height(height)
{
  GLuint texs[2];
  glCreateFramebuffers(1, &_id);
  glCreateTextures(GL_TEXTURE_2D, 2, texs);
  _color_id = texs[0];
  _depth_id = texs[1];

  printf("Creating frame buffer of size %d %d \n", width, height);


  GLenum fmt = (linear) ? GL_RGB8 : GL_SRGB_ALPHA;
  glTextureStorage2D(_color_id, 1, fmt, width, height);
  glTextureParameteri(_color_id, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTextureParameteri(_color_id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glTextureStorage2D(_depth_id, 1, depth_format, width, height);

  glNamedFramebufferTexture(_id, GL_COLOR_ATTACHMENT0, _color_id, 0);
  glNamedFramebufferTexture(_id, GL_DEPTH_ATTACHMENT, _depth_id, 0);

  if (!linear) {
    int p;
    glGetNamedFramebufferAttachmentParameteriv(_id, GL_COLOR_ATTACHMENT0,
                                               GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING,
                                               &p);
    assert(p == GL_SRGB);
  }
}

GlFrameBuffer::~GlFrameBuffer() {
  glDeleteFramebuffers(1, &_id);
  GLuint textures[2] = {_color_id, _depth_id};
  glDeleteTextures(2, textures);
}

void GlFrameBuffer::bind(GLenum target) const {
  glBindFramebuffer(target, _id);
}

void GlFrameBuffer::blit_entirely(GLuint dst) const {
  glBlitNamedFramebuffer(_id, dst,
                         0, 0, _width, _height,
                         0, 0, _width, _height,
                         GL_COLOR_BUFFER_BIT,
                         GL_LINEAR);
}

////////////////////////////////////////////////////////////////////////////////
// GlBuffer
////////////////////////////////////////////////////////////////////////////////

template<typename T>
class GlBuffer {
  const static int NUM_BUFFERS = 1;
private:
  /** Name of the buffer under management */
  GLuint _ids[NUM_BUFFERS];
  /** Current buffer */
  int _buffer;
  /** The number of bytes allocated in that buffer */
  size_t _capacities[NUM_BUFFERS];
  /** CPU-side storage for all the information */
  std::vector<T> _storage;

  friend GlVAO<T>;
  template<typename F> friend class ParticleSystem;
public:
  GlBuffer() {
    glCreateBuffers(NUM_BUFFERS, _ids);
    _buffer = 0;
    _storage.reserve(1); // Make sure there is room for at least 1 element
    for (auto i = 0; i < NUM_BUFFERS; ++i) {
      _capacities[i] = _storage.capacity() * sizeof(T);
      glNamedBufferData(_ids[i], _capacities[i], NULL, GL_STREAM_DRAW);
    }
  }

  GlBuffer(GlBuffer & other) = delete;

  GlBuffer(GlBuffer && other) {
    _ids = other._ids;
    _capacities = other._capacities;
    _storage = std::move(other._storage);
  }

  GlBuffer & operator= (GlBuffer & other) = delete;

  GlBuffer & operator= (GlBuffer && other) {
    _ids = other._ids;
    _capacities = other._capacities;
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
    size_t bsize = _storage.size() * sizeof(T);
    if (_capacities[_buffer] < bsize) {
      _capacities[_buffer] = bsize;
      glNamedBufferData(_ids[_buffer], _capacities[_buffer], NULL, GL_STREAM_DRAW);
    }
    glNamedBufferSubData(_ids[_buffer], 0, bsize, _storage.data());
    _buffer = (_buffer + 1) % NUM_BUFFERS;
  }

  void clear() {
    _storage.clear();
  }

  typename std::vector<T>::size_type num_elems() const {
    return _storage.size();
  }
};

//////////////////////////////////////////////////////////////////////////////
// GlShader
//////////////////////////////////////////////////////////////////////////////

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

GlShader::GlShader(GlShader && other) {
  _prog_id = other._prog_id;
}

void GlShader::bind() {
  glUseProgram(_prog_id);
}

void GlShader::uniform(GLint location, GLboolean transpose, float matrix[16]) {
  glProgramUniformMatrix4fv(_prog_id, location, 1, transpose, matrix);
}

////////////////////////////////////////////////////////////////////////////////
// GlVAO
////////////////////////////////////////////////////////////////////////////////

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
  glVertexArrayVertexBuffer(_id, buffer_index, buffer->_ids[buffer->_buffer], offset, stride);
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

////////////////////////////////////////////////////////////////////////////////
// Particle system
////////////////////////////////////////////////////////////////////////////////
const char * particle_vert_shader =
  "#version 450\n"
  "layout(location=0) in vec4 position_life;\n"
  "layout(location=1) in vec4 in_color;\n"
  "layout(location=0) uniform mat4 transform;\n"
  "out float life;\n"
  "out vec4 cross_color;\n"
  "void main() {"
  "  gl_Position = transform * vec4(position_life.xyz, 1);\n"
  "  gl_PointSize = 3.f;\n"
  "  cross_color = in_color;\n"
  "  life = position_life.w;\n"
  "}\n";

static const char * particle_frag_shader =
  "#version 450\n"
  "out vec4 color;\n"
  "in vec4 cross_color;\n"
  "in float life;\n"
  "void main() {\n"
  "  color = cross_color;\n"
  "}\n";

struct __attribute__((packed)) particle {
  float x, y, z, t;
  float vx, vy, vz;
  struct color c;
};

template<typename UpdateFunc>
class ParticleSystem {
private:
  std::shared_ptr<GlBuffer<particle>> _particles;
  std::shared_ptr<GlVAO<particle>> _vao;
  GlShader _shader;
  UpdateFunc _update;
  float _max_age;
public:
  ParticleSystem(UpdateFunc f, float max_age);
  ParticleSystem(const ParticleSystem & other) = delete;
  ParticleSystem(const ParticleSystem && other);
  ParticleSystem & operator=(ParticleSystem & other) = delete;
  ParticleSystem & operator=(ParticleSystem && other);
  ~ParticleSystem();

  void add_particle(float x, float y, float vx, float vy, float t, const color & c);
  void do_update(float dt);
  void render(float transform[16]);
};

template<typename F>
ParticleSystem<F>::ParticleSystem(F f, float max_age)
  : _particles(new GlBuffer<particle>()),
    _vao(new GlVAO<particle>(_particles)),
    _shader(std::string(particle_vert_shader), std::string(particle_frag_shader)),
    _update(f),
    _max_age(max_age)
{
  _vao->enable_attrib(POSITION_LOCATION);
  _vao->enable_attrib(COLOR_LOCATION);

  _vao->bind_attrib(POSITION_LOCATION, 0);
  _vao->bind_attrib(COLOR_LOCATION, 0);

  _vao->attrib_format(POSITION_LOCATION, 4, GL_FLOAT, GL_FALSE, offsetof(particle, x));
  _vao->attrib_format(COLOR_LOCATION, 4, GL_FLOAT, GL_FALSE, offsetof(particle, c));
}

template<typename F>
ParticleSystem<F>::~ParticleSystem() {}

template<typename F>
void ParticleSystem<F>::add_particle(float x, float y, float vx, float vy, float t, const color & c) {
  particle np = {
    .x = x, .y = y, .t = t, .c = c,
    .vx = vx, .vy = vy,
  };
  if (_particles->_storage.size() > 32192) {
    int ridx = rand() % 32192;
    _particles->_storage[ridx] = np;
  } else {
    _particles->_storage.push_back(np);
  }
}

template<typename F>
void ParticleSystem<F>::do_update(float dt) {
  dt /= _max_age;
  auto & store = _particles->_storage;
  for (auto i = 0; i < store.size(); i++) {
    _update(store[i], dt);
    if (store[i].t > 1.f) {
      store[i] = store[store.size() - 1];
      store.pop_back();
    }
  }
}

template<typename F>
void ParticleSystem<F>::render(float transform[16]) {
  if (_particles->num_elems() == 0) {
    return;
  }
  _vao->bind_buffer(_particles, 0, 0, sizeof(particle));

  _particles->sync();
  _shader.bind();
  _vao->bind();
  _shader.uniform(TRANSFORM_LOCATION, GL_TRUE, transform);

  glDrawArrays(GL_POINTS, 0, _particles->num_elems());
}

////////////////////////////////////////////////////////////////////////////////
// atlas
////////////////////////////////////////////////////////////////////////////////

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

  void bind_texture(int binding) {
    (void) binding;
    glBindTexture(GL_TEXTURE_2D, tex.GetID());
  }
};

////////////////////////////////////////////////////////////////////////////////
// FontBatch
////////////////////////////////////////////////////////////////////////////////

class FontBatch {
private:
  std::shared_ptr<GlBuffer<vertex>> _verts;
  std::shared_ptr<GlVAO<vertex>> _vert_vao;
public:
  FontBatch();
  FontBatch(const FontBatch & other) = delete;
  FontBatch(FontBatch && other);
  FontBatch & operator=(const FontBatch & other) = delete;
  FontBatch & operator=(FontBatch && other);
  ~FontBatch();

  void enqueue_glyph(const glyph_spec * const spec);
  void render();
};

FontBatch::FontBatch() :
  _verts(new GlBuffer<vertex>()),
  _vert_vao(new GlVAO<vertex>(_verts)) {
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

FontBatch::~FontBatch() {}

void FontBatch::enqueue_glyph(const glyph_spec * const spec) {
  struct glyph_render_params rps;
  spec->font->glyph_render_params(spec->glyph, rps);

  float pixel_width = ATLAS_SIZE * rps.uvs.size.x;
  float pixel_height = ATLAS_SIZE * rps.uvs.size.y;
  float base_x = rps.offset.x + static_cast<float>(spec->x);
  float base_y = -rps.offset.y + static_cast<float>(spec->y);

  if (pixel_width < 1 || pixel_height < 1) {
    // Don't enqueue tiny shapes
    return;
  }

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

void FontBatch::render() {
  _vert_vao->bind_buffer(_verts, 0, 0, sizeof(vertex));

  _verts->sync();
  _vert_vao->bind();
  glDrawArrays(GL_TRIANGLES, 0, _verts->num_elems());
  _verts->clear();
}

////////////////////////////////////////////////////////////////////////////////
//  FBBlitJob
////////////////////////////////////////////////////////////////////////////////

class FBBlitJob {
private:
  std::shared_ptr<GlShader> _shader;
  std::shared_ptr<GlBuffer<vertex>> _verts;
  std::shared_ptr<GlVAO<vertex>> _vert_vao;
public:
  FBBlitJob(std::shared_ptr<GlShader> shader);
  FBBlitJob(const FBBlitJob & other) = delete;
  FBBlitJob(const FBBlitJob && other);
  FBBlitJob& operator=(const FBBlitJob & other) = delete;
  FBBlitJob& operator=(const FBBlitJob && other) = delete;
  ~FBBlitJob();

  void do_blit(GLuint backing_texture);
};

FBBlitJob::FBBlitJob(std::shared_ptr<GlShader> shader)
  : _shader(shader),
    _verts(new GlBuffer<vertex>()),
    _vert_vao(new GlVAO<vertex>(_verts))
{
  _vert_vao->enable_attrib(POSITION_LOCATION);
  _vert_vao->enable_attrib(UV_LOCATION);
  _vert_vao->enable_attrib(COLOR_LOCATION);

  _vert_vao->bind_attrib(POSITION_LOCATION, 0);
  _vert_vao->bind_attrib(UV_LOCATION, 0);
  _vert_vao->bind_attrib(COLOR_LOCATION, 0);

  _vert_vao->attrib_format(POSITION_LOCATION, 3, GL_FLOAT, GL_FALSE, offsetof(vertex, pos));
  _vert_vao->attrib_format(UV_LOCATION, 2, GL_FLOAT, GL_FALSE, offsetof(vertex, texcoords));
  _vert_vao->attrib_format(COLOR_LOCATION, 4, GL_FLOAT, GL_FALSE, offsetof(vertex, c));

  vertex lverts[4] = {
    {
      .pos = {.x = -1.f, .y = -1.f, .z = 0.f },
      .texcoords = {.x = 0.f, .y = 0.f },
      .c = { .r = 1.f, .g = 1.f, .b = 1.f }
    },
    {
      .pos = {.x =  1.f, .y =  1.f, .z = 0.f },
      .texcoords = {.x = 1.f, .y = 1.f },
      .c = { .r = 1.f, .g = 1.f, .b = 1.f }
    },
    {
      .pos = {.x = -1.f, .y =  1.f, .z = 0.f },
      .texcoords = {.x = 0.f, .y = 1.f },
      .c = { .r = 1.f, .g = 1.f, .b = 1.f }
    },
    {
      .pos = {.x =  1.f, .y = -1.f, .z = 0.f },
      .texcoords = {.x = 1.f, .y = 0.f },
      .c = { .r = 1.f, .g = 1.f, .b = 1.f }
    },
  };

  _verts->push_elements(lverts, 3);
  lverts[2] = lverts[3];
  _verts->push_elements(lverts, 3);

  _verts->sync();
}

FBBlitJob::~FBBlitJob() {
}

void FBBlitJob::do_blit(GLuint backing_texture) {
  _vert_vao->bind_buffer(_verts, 0, 0, sizeof(vertex));

  glBindTextureUnit(TEXTURE_BINDING, backing_texture);
  _shader->bind();
  float transform[16] = {
    1.f, 0.f, 0.f, 0.f,
    0.f, 1.f, 0.f, 0.f,
    0.f, 0.f, 1.f, 0.f,
    0.f, 0.f, 0.f, 1.f,
  };
  _shader->uniform(TRANSFORM_LOCATION, GL_TRUE, transform);

  _vert_vao->bind();
  glDrawArrays(GL_TRIANGLES, 0, _verts->num_elems());
}

////////////////////////////////////////////////////////////////////////////////
//  RectJob
////////////////////////////////////////////////////////////////////////////////

class RectJob {
private:
  std::shared_ptr<GlShader> _shader;
  std::shared_ptr<GlBuffer<vertex>> _verts;
  std::shared_ptr<GlVAO<vertex>> _vert_vao;
public:
  RectJob(std::shared_ptr<GlShader> shader);
  RectJob(const RectJob & other) = delete;
  RectJob(const RectJob && other);
  RectJob& operator=(const RectJob & other) = delete;
  RectJob& operator=(const RectJob && other) = delete;
  ~RectJob();

  void draw_rect(const struct color * const c, int x, int y, int w, int h);
  void render(float transform[16]);
};

RectJob::RectJob(std::shared_ptr<GlShader> shader)
  : _shader(shader),
    _verts(new GlBuffer<vertex>()),
    _vert_vao(new GlVAO<vertex>(_verts))
{
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

RectJob::~RectJob() {
}

void RectJob::render(float transform[16]) {
  _vert_vao->bind_buffer(_verts, 0, 0, sizeof(vertex));
  _verts->sync();
  _shader->bind();
  _shader->uniform(TRANSFORM_LOCATION, GL_TRUE, transform);
  _vert_vao->bind();
  glDrawArrays(GL_TRIANGLES, 0, _verts->num_elems());
  _verts->clear();
}

void RectJob::draw_rect(const struct color * const c, int xi, int yi, int w, int h) {
  if (w == 0 || h == 0) {
    // Don't enqueue empty rectangles
    return;
  }
  float x = static_cast<float>(xi);
  float y = static_cast<float>(yi);
  vertex lverts[4] = {
    {
      .pos = { .x = x, .y = y, .z = 0 },
      .texcoords = { .x = 0, .y = 0 },
      .c = *c
    },
    {
      .pos = { .x = x + w, .y = y + h, .z = 0},
      .texcoords = { .x = 1, .y = 1 },
      .c = *c
    },
    {
      .pos = { .x = x, .y = y + h, .z = 0 },
      .texcoords = { .x = 0, .y = 1 },
      .c = *c
    },
    {
      .pos = { .x = x + w, .y = y, .z = 0 },
      .texcoords = { .x = 1, .y = 0 },
      .c = *c
    },
  };

  _verts->push_elements(lverts, 3);
  lverts[2] = lverts[3];
  _verts->push_elements(lverts, 3);
}

////////////////////////////////////////////////////////////////////////////////
//  render_context
////////////////////////////////////////////////////////////////////////////////

void flame_curve(float f, color & c) {
  // t0 = 1
  // t1 = f
  float t2 = 2 * f * f - 1;
  float t3 = f * (2 * t2 - 1); // 2 f t2 -  t1
  float t4 = 2 * f * t3 - t2;

  const float rc[] = { -7.42828172, 13.49066809, -9.02608052,  3.68452288, -0.71117265 };
  const float gc[] = { -4.56657944, 8.57004895, -6.62752371, 3.5393551, -0.99186063 };
  const float bc[] = { 8.60556488, -14.57317637,  8.7124878, -3.42279269,  0.7211981 };
  c.r = rc[0] + f * rc[1] + t2 * rc[2] + t3 * rc[3] + t4 * rc[4];
  c.g = gc[0] + f * gc[1] + t2 * gc[2] + t3 * gc[3] + t4 * gc[4];
  c.b = bc[0] + f * bc[1] + t2 * bc[2] + t3 * bc[3] + t4 * bc[4];
  c.a = 1.f - f;
}

void basic_update(particle & p, float dt) {
  p.t += dt;
  p.x += dt * p.vx;
  p.y += dt * p.vy;
  p.vy += 100.f * dt;
  flame_curve(p.t, p.c);
}

struct render_context {
  GlShader _shader;
  color _clear_color;
  std::unique_ptr<GlFrameBuffer> _fb;
  std::unique_ptr<GlFrameBuffer> _particle_fb;
  std::unordered_map<struct atlas*, FontBatch> _text_batches;
  int _win_w;
  int _win_h;
  ParticleSystem<std::function<void(particle&, float)>> _parts;

  FBBlitJob _fb_blitter;
  FBBlitJob _particle_blitter;
  RectJob _rect_job;

  void set_size(int w, int h);
  void set_y_nudge(int y);
  void do_render();
  void render_rune(const glyph_spec * spec);
  void set_clear_color(const color & c);

  render_context();
};

void render_context::do_render() {
  ///
  // Update step
  ///
  _parts.do_update(0.16f);

  ///
  // Actual render step
  ///

  // Transform matrix in row major order
  float transform[16] = {
    2.f / _win_w, 0.f, 0.f, -1.f,
    0.f, -2.f / _win_h, 0.f, 1.f,
    0.f, 0.f, 1.f, 0.f,
    0.f, 0.f, 0.f, 1.f,
  };

  // Render particles
  _particle_fb->bind(GL_DRAW_FRAMEBUFFER);
  glViewport(0, 0, _win_w / PARTICLE_FB_SCALE, _win_h / PARTICLE_FB_SCALE);
  glClearColor(0.f, 0.f, 0.f, 0.f);
  glClear(GL_COLOR_BUFFER_BIT);
  _parts.render(transform);

  // Bind main framebuffer
  _fb->bind(GL_DRAW_FRAMEBUFFER);
  glClearColor(_clear_color.r, _clear_color.g, _clear_color.b, _clear_color.a);
  glClear(GL_COLOR_BUFFER_BIT);

  // Update viewport to correct size
  glViewport(0, 0, _win_w, _win_h);

  // Render rectangles
  _rect_job.render(transform);

  // Render fonts
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  {
    _shader.bind();
    _shader.uniform(TRANSFORM_LOCATION, GL_TRUE, transform);

    _shader.bind();
    for (auto & kv : _text_batches) {
      kv.first->bind_texture(TEXTURE_BINDING);
      kv.second.render();
    }

  }

  // Blit particles
  _particle_blitter.do_blit(_particle_fb->get_main_color());
  glDisable(GL_BLEND);

  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  _fb_blitter.do_blit(_fb->get_main_color());
}

void render_context::render_rune(const glyph_spec * spec) {
  auto kv = _text_batches.find(spec->font);
  if (kv == _text_batches.end()) {
    kv = _text_batches.emplace(std::piecewise_construct,
                               std::forward_as_tuple(spec->font),
                               std::forward_as_tuple()).first;
  }
  if (spec->dirty) {
    color c = *spec->c;
    c.a = 0.5f;
    for (int i = 0; i < 512; ++i) {
      const float jitter = -50.f * (rand() /(float) RAND_MAX) - 10.f;
      float x_jitter = 2.f * rand() / (float) RAND_MAX - 1.f;
      float y_jitter = sqrt(1 - x_jitter * x_jitter);
      float t_jitter = 0.3f * (rand() / (float) RAND_MAX);
      _parts.add_particle(spec->x, spec->y, jitter * x_jitter - jitter / 2, -50.f + jitter * y_jitter, t_jitter, c);
    }
  }
  kv->second.enqueue_glyph(spec);
}

void render_context::set_clear_color(const color & c) {
  _clear_color = c;
}

void render_context::set_size(int w, int h) {
  _win_w = w;
  _win_h = h;
  _fb.reset(new GlFrameBuffer(w, h, false, GL_DEPTH_COMPONENT24));
  _particle_fb.reset(new GlFrameBuffer(w / PARTICLE_FB_SCALE, h / PARTICLE_FB_SCALE, false, GL_DEPTH_COMPONENT24));
}

void render_context::set_y_nudge(int y) {
  printf("Y nudged %d\n", y);
}

render_context::render_context()
  : _shader(std::string(vert_shader), std::string(frag_shader)),
    _fb(new GlFrameBuffer(1, 1, false, GL_DEPTH_COMPONENT24)),
    _particle_fb(new GlFrameBuffer(1, 1, false, GL_DEPTH_COMPONENT24)),
    _win_w(1),
    _win_h(1),
    _parts(basic_update, 16.f),
    _fb_blitter(std::make_shared<GlShader>(std::string(vert_shader), std::string(framebuffer_frag_shader))),
    _particle_blitter(std::make_shared<GlShader>(std::string(vert_shader), std::string(particle_blit_shader))),
    _rect_job(std::make_shared<GlShader>(std::string(vert_shader), std::string(color_shader)))
{
  glEnable(GL_FRAMEBUFFER_SRGB);
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
  printf("Render resize to %d %d\n", w, h);
  rc->set_size(w, h);
}

void render_set_y_nudge(struct render_context * rc, int y) {
  rc->set_y_nudge(y);
}

void render_set_clear_color(struct render_context * rc, struct color * c) {
  rc->set_clear_color(*c);
}

void render_do_render(struct render_context * rc) {
  rc->do_render();
}

struct atlas * atlas_create_from_face(FT_Face f) {
  return new atlas(f);
}

struct atlas * atlas_create_from_pattern(FT_Library lib, FcPattern * pat, FT_UInt size) {
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
  if (FT_New_Face(lib, file_name, file_index, &f) != 0) {
    fputs("st: failed to open font file\n", stderr);
    return nullptr;
  }
  FT_Set_Pixel_Sizes(f, 0, size);
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

void render_rect(struct render_context * rc, const struct color * const c, int x, int y, int w, int h) {
  rc->_rect_job.draw_rect(c, x, y, w, h);
}
