#include <GL/glut.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>

namespace fs = std::filesystem;

struct Point3D { float x, y, z; };

// -------- state --------
static std::vector<std::vector<Point3D>> frames;
static std::vector<std::string>          frame_names;
static int   current_frame = 0;
static bool  playing       = true;
static int   fps           = 30;     // playback rate
static int   timer_ms      = 33;     // 1000/fps

// camera
static float angleX = 15.0f, angleY = 0.0f, zoom = 15.0f;
static float zoom_default = 15.0f;
static float cx = 0, cy = 0, cz = 0;
static int   lastMouseX = -1, lastMouseY = -1;

// =============================================================================
// File loading
// =============================================================================
static bool load_one_file(const std::string& filename,
                          std::vector<Point3D>& out)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "WARN: could not open " << filename << '\n';
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        float u, v, x, y, z;
        if (ss >> u >> v >> x >> y >> z) out.push_back({x, y, z});
    }
    return true;
}

static void compute_centroid_and_fit(const std::vector<Point3D>& pts) {
    if (pts.empty()) return;
    double sx = 0, sy = 0, sz = 0;
    for (auto& p : pts) { sx += p.x; sy += p.y; sz += p.z; }
    cx = float(sx / pts.size());
    cy = float(sy / pts.size());
    cz = float(sz / pts.size());
    float r2 = 0.0f;
    for (auto& p : pts) {
        float dx = p.x - cx, dy = p.y - cy, dz = p.z - cz;
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 > r2) r2 = d2;
    }
    zoom_default = std::max(10.0f, std::sqrt(r2) * 1.6f);
    zoom = zoom_default;
}

// =============================================================================
// Drawing
// =============================================================================
static void drawAxes() {
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glColor3f(1, 0, 0); glVertex3f(0,0,0); glVertex3f(2,0,0);
    glColor3f(0, 1, 0); glVertex3f(0,0,0); glVertex3f(0,2,0);
    glColor3f(0, 0.5f, 1); glVertex3f(0,0,0); glVertex3f(0,0,2);
    glEnd();
}

static void drawHUD() {
    // Switch to 2-D ortho.
    int w = glutGet(GLUT_WINDOW_WIDTH);
    int h = glutGet(GLUT_WINDOW_HEIGHT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, w, 0, h);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glColor3f(1, 1, 0);

    char buf[256];
    if (!frames.empty()) {
        std::snprintf(buf, sizeof(buf),
            "Frame %d/%zu   FPS %d   Pts %zu   %s   [%s]",
            current_frame + 1, frames.size(), fps,
            frames[current_frame].size(),
            playing ? "PLAY" : "PAUSE",
            (current_frame < (int)frame_names.size()
                 ? frame_names[current_frame].c_str() : ""));
    } else {
        std::snprintf(buf, sizeof(buf), "no frames loaded");
    }

    glRasterPos2i(10, h - 20);
    for (const char* p = buf; *p; ++p)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *p);

    const char* help = "[Space] play/pause   [<- ->] step   [+ -] fps   "
                       "[r] reset frame   [0] reset view";
    glRasterPos2i(10, 10);
    for (const char* p = help; *p; ++p)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *p);

    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

static void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    gluLookAt(cx, cy + 2.0f, cz - zoom,
              cx, cy, cz,
              0.0f, 1.0f, 0.0f);

    glTranslatef(cx, cy, cz);
    glRotatef(angleX, 1, 0, 0);
    glRotatef(angleY, 0, 1, 0);
    glTranslatef(-cx, -cy, -cz);

    drawAxes();

    if (!frames.empty()) {
        glPointSize(3.0f);
        glColor3f(1, 1, 1);
        glBegin(GL_POINTS);
        for (const auto& p : frames[current_frame])
            glVertex3f(p.x, p.y, p.z);
        glEnd();
    }

    drawHUD();
    glutSwapBuffers();
}

static void reshape(int w, int h) {
    if (h == 0) h = 1;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, double(w)/h, 0.1, 1000.0);
    glMatrixMode(GL_MODELVIEW);
}

// =============================================================================
// Input handlers
// =============================================================================
static void mouseMove(int x, int y) {
    if (lastMouseX >= 0 && lastMouseY >= 0) {
        angleY += (x - lastMouseX) * 0.5f;
        angleX += (y - lastMouseY) * 0.5f;
    }
    lastMouseX = x; lastMouseY = y;
    glutPostRedisplay();
}

static void mouseButton(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
        lastMouseX = x; lastMouseY = y;
        if (button == 3) zoom -= 1.0f;
        if (button == 4) zoom += 1.0f;
    } else {
        lastMouseX = -1; lastMouseY = -1;
    }
    glutPostRedisplay();
}

static void keyboard(unsigned char key, int, int) {
    switch (key) {
        case 'w': case 'W': zoom -= 1.0f; break;
        case 's': case 'S': zoom += 1.0f; break;
        case ' ': playing = !playing; break;
        case '+': case '=':
            fps = std::min(120, fps + 5); timer_ms = 1000 / fps; break;
        case '-': case '_':
            fps = std::max(5, fps - 5);   timer_ms = 1000 / fps; break;
        case 'r': case 'R':
            current_frame = 0; break;
        case '0':
            angleX = 15.0f; angleY = 0.0f; zoom = zoom_default; break;
        case 27: std::exit(0);
    }
    glutPostRedisplay();
}

static void specialKey(int key, int, int) {
    if (frames.empty()) return;
    switch (key) {
        case GLUT_KEY_LEFT:
            playing = false;
            current_frame = (current_frame - 1 + frames.size()) % frames.size();
            break;
        case GLUT_KEY_RIGHT:
            playing = false;
            current_frame = (current_frame + 1) % frames.size();
            break;
    }
    glutPostRedisplay();
}

static void timer(int) {
    if (playing && !frames.empty())
        current_frame = (current_frame + 1) % frames.size();
    glutPostRedisplay();
    glutTimerFunc(timer_ms, timer, 0);
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <points_dir | single_file.txt>\n";
        return 1;
    }

    fs::path arg = argv[1];
    if (fs::is_directory(arg)) {
        std::vector<fs::path> files;
        for (auto& e : fs::directory_iterator(arg)) {
            if (!e.is_regular_file()) continue;
            if (e.path().extension() == ".txt") files.push_back(e.path());
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) {
            std::cerr << "No .txt files in " << arg.string() << '\n';
            return 1;
        }
        frames.reserve(files.size());
        frame_names.reserve(files.size());
        for (auto& f : files) {
            std::vector<Point3D> pts;
            if (load_one_file(f.string(), pts)) {
                frames.push_back(std::move(pts));
                frame_names.push_back(f.filename().string());
            }
        }
        std::cerr << "Loaded " << frames.size() << " frames.\n";
        if (!frames.empty()) compute_centroid_and_fit(frames[0]);
    } else {
        std::vector<Point3D> pts;
        if (!load_one_file(arg.string(), pts)) return 1;
        frames.push_back(std::move(pts));
        frame_names.push_back(arg.filename().string());
        compute_centroid_and_fit(frames[0]);
        playing = false;
    }
    std::cerr << "Centroid: (" << cx << ", " << cy << ", " << cz << ")\n";

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1000, 700);
    glutCreateWindow("IPM Point Cloud Viewer");

    glEnable(GL_DEPTH_TEST);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMotionFunc(mouseMove);
    glutMouseFunc(mouseButton);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKey);
    glutTimerFunc(timer_ms, timer, 0);

    glutMainLoop();
    return 0;
}
