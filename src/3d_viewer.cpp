#include <GL/glut.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Structure to hold our 3D coordinates
struct Point3D {
    float x, y, z;
};

std::vector<Point3D> pointCloud;

// Camera and interaction variables
float angleX = 15.0f; // Start with a slight pitch (matches your 15 deg camera pitch)
float angleY = 0.0f;
float zoom = 15.0f;
int lastMouseX = -1;
int lastMouseY = -1;

// Center of the point cloud for camera targeting
float cx = 0, cy = 0, cz = 0;

void loadPointCloud(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::string line;
    float sumX = 0, sumY = 0, sumZ = 0;

    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream ss(line);
        float u, v, x, y, z;
        
        // Extract the 5 columns: pixel_u, pixel_v, Xc, Yc, Zc
        if (ss >> u >> v >> x >> y >> z) {
            pointCloud.push_back({x, y, z});
            sumX += x;
            sumY += y;
            sumZ += z;
        }
    }
    file.close();

    // Calculate centroid to center the camera automatically
    if (!pointCloud.empty()) {
        cx = sumX / pointCloud.size();
        cy = sumY / pointCloud.size();
        cz = sumZ / pointCloud.size();
        std::cout << "Loaded " << pointCloud.size() << " points." << std::endl;
        std::cout << "Cloud Centroid: (" << cx << ", " << cy << ", " << cz << ")" << std::endl;
    }
}

void drawAxes() {
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    // X axis (Red)
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(2.0f, 0.0f, 0.0f);
    // Y axis (Green)
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 2.0f, 0.0f);
    // Z axis (Blue)
    glColor3f(0.0f, 0.5f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 2.0f);
    glEnd();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // Position camera looking at the centroid of the point cloud
    gluLookAt(cx, cy + 2.0f, cz - zoom,  // Eye position
              cx, cy, cz,                // Look-at point
              0.0f, 1.0f, 0.0f);         // Up vector

    // Apply mouse rotations
    glTranslatef(cx, cy, cz);
    glRotatef(angleX, 1.0f, 0.0f, 0.0f);
    glRotatef(angleY, 0.0f, 1.0f, 0.0f);
    glTranslatef(-cx, -cy, -cz);

    drawAxes();

    // Render the Point Cloud
    glPointSize(3.0f); // Make points slightly larger
    glColor3f(1.0f, 1.0f, 1.0f); // White points
    
    glBegin(GL_POINTS);
    for (const auto& p : pointCloud) {
        glVertex3f(p.x, p.y, p.z);
    }
    glEnd();

    glutSwapBuffers();
}

void reshape(int w, int h) {
    if (h == 0) h = 1;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (float)w / h, 0.1f, 1000.0f);
    glMatrixMode(GL_MODELVIEW);
}

void mouseMove(int x, int y) {
    if (lastMouseX >= 0 && lastMouseY >= 0) {
        angleY += (x - lastMouseX) * 0.5f;
        angleX += (y - lastMouseY) * 0.5f;
    }
    lastMouseX = x;
    lastMouseY = y;
    glutPostRedisplay();
}

void mouseButton(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
        lastMouseX = x;
        lastMouseY = y;
        
        // Scroll wheel zooming
        if (button == 3) zoom -= 1.0f; // Scroll up
        if (button == 4) zoom += 1.0f; // Scroll down
    } else {
        lastMouseX = -1;
        lastMouseY = -1;
    }
    glutPostRedisplay();
}

// Keyboard controls for zooming if scroll wheel isn't detected
void keyboard(unsigned char key, int x, int y) {
    if (key == 'w' || key == 'W') zoom -= 1.0f;
    if (key == 's' || key == 'S') zoom += 1.0f;
    if (key == 27) exit(0); // ESC to quit
    glutPostRedisplay();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_points_file.txt>" << std::endl;
        return 1;
    }

    loadPointCloud(argv[1]);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("IPM Point Cloud Viewer");

    glEnable(GL_DEPTH_TEST);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMotionFunc(mouseMove);
    glutMouseFunc(mouseButton);
    glutKeyboardFunc(keyboard);

    glutMainLoop();
    return 0;
}