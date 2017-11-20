/* See LICENSE for license details. */

/* X modifiers */
#define XK_ANY_MOD    UINT_MAX
#define XK_NO_MOD     0
#define XK_SWITCH_MOD (1<<13)

typedef XftGlyphFontSpec GlyphFontSpec;

/** Redraws the whole screen */
void draw(void);
/** Draws the specified region

   @param x1 is the min x coordinate
   @param y1 is the min y coordinate
   @param x2 is the max x coordinate
   @param y2 is the may y coordinate
*/
void drawregion(int x1, int y1, int x2, int y2);
/** Runs the main loop */
void run(void);

/** Make a ding noise

   @param vol Volume to play bell at
*/
void xbell(int vol);

/******************************************************************************\
 * Copy paste buffer handling
\******************************************************************************/
void xclipcopy(void);
void xclippaste(void);
void xselpaste(void);
void xsetsel(char *, Time);

/** Set all the XWMHints */
void xhints(void);
/** Open the display and our window */
void xinit(void);
/** Load colors from the X server */
void xloadcols(void);
/** Name colors */
int xsetcolorname(int, const char *);

/** Load the font */
void xloadfonts(char *, double);
/** Unload fonts */
void xunloadfonts(void);

/** Set X11 environment variables */
void xsetenv(void);
/** Set our window title */
void xsettitle(char *);
/** Pass a nonzero value to enable pointer motion listening */
void xsetpointermotion(int);
/** Request the users's interaction via the X11 urgency flags */
void xseturgency(int);
/** Called when a resize occurs */
void xresize(int, int);
/** Get the window ID */
unsigned long xwinid(void);
