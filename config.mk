# st version
VERSION = 0.7

# Customize below to fit your system

# paths
PREFIX = /usr/local
MANPREFIX = $(PREFIX)/share/man

X11INC = /usr/X11R6/include
X11LIB = /usr/X11R6/lib

# includes and libs
INCS = -I$(X11INC) \
			 -isystem ./include \
       `pkg-config --cflags fontconfig` \
       `pkg-config --cflags gl` \
       `pkg-config --cflags freetype2`
LIBS = -L$(X11LIB) -lm -lrt -lX11 -lutil -lXft -ldl \
       `pkg-config --libs fontconfig` \
       `pkg-config --libs gl` \
       `pkg-config --libs freetype2`

# flags
CPPFLAGS = -DVERSION=\"$(VERSION)\" -D_XOPEN_SOURCE=600
STDCXXFLAGS = $(CXXFLAGS) $(INCS) $(CPPFLAGS) -std=c++17
STCFLAGS = $(INCS) $(CPPFLAGS) $(CFLAGS)
STLDFLAGS = $(LIBS) $(LDFLAGS)

# compiler and linker
# CC = c99

