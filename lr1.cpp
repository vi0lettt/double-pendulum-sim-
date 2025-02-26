#include <iostream>
#include <vector>
#include <cmath>
#include <GL/gl.h>
#include <GL/glu.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

using namespace std;

const double g = 9.81;
const double m1 = 1.0, m2 = 3.0;
const double l1 = 100.0, l2 = 100.0;
const double dt = 0.05;
const int width = 800, height = 600;

struct State {
    double theta1, theta2, omega1, omega2;
};

State s = {M_PI / 2, M_PI / 2, 0.0, 0.0};

vector<pair<double, double>> trajectory;  

double f1(double theta1, double theta2, double omega1, double omega2) {
    double delta_theta = theta1 - theta2;
    double denom = l1 * (2 * m1 + m2 - m2 * cos(2 * delta_theta));
    return (-g * (2 * m1 + m2) * sin(theta1) - m2 * g * sin(theta1 - 2 * theta2)
            - 2 * sin(delta_theta) * m2 * (omega2 * omega2 * l2 + omega1 * omega1 * l1 * cos(delta_theta))) / denom;
}

double f2(double theta1, double theta2, double omega1, double omega2) {
    double delta_theta = theta1 - theta2;
    double denom = l2 * (2 * m1 + m2 - m2 * cos(2 * delta_theta));
    return (2 * sin(delta_theta) * (omega1 * omega1 * l1 * (m1 + m2) + g * (m1 + m2) * cos(theta1)
                                    + omega2 * omega2 * l2 * m2 * cos(delta_theta))) / denom;
}



void runge_kutta4(State &s) {
    double k1_theta1 = dt * s.omega1;
    double k1_theta2 = dt * s.omega2;
    double k1_omega1 = dt * f1(s.theta1, s.theta2, s.omega1, s.omega2);
    double k1_omega2 = dt * f2(s.theta1, s.theta2, s.omega1, s.omega2);

    double k2_theta1 = dt * (s.omega1 + 0.5 * k1_omega1);
    double k2_theta2 = dt * (s.omega2 + 0.5 * k1_omega2);
    double k2_omega1 = dt * f1(s.theta1 + 0.5 * k1_theta1, s.theta2 + 0.5 * k1_theta2, s.omega1 + 0.5 * k1_omega1, s.omega2 + 0.5 * k1_omega2);
    double k2_omega2 = dt * f2(s.theta1 + 0.5 * k1_theta1, s.theta2 + 0.5 * k1_theta2, s.omega1 + 0.5 * k1_omega1, s.omega2 + 0.5 * k1_omega2);

    double k3_theta1 = dt * (s.omega1 + 0.5 * k2_omega1);
    double k3_theta2 = dt * (s.omega2 + 0.5 * k2_omega2);
    double k3_omega1 = dt * f1(s.theta1 + 0.5 * k2_theta1, s.theta2 + 0.5 * k2_theta2, s.omega1 + 0.5 * k2_omega1, s.omega2 + 0.5 * k2_omega2);
    double k3_omega2 = dt * f2(s.theta1 + 0.5 * k2_theta1, s.theta2 + 0.5 * k2_theta2, s.omega1 + 0.5 * k2_omega1, s.omega2 + 0.5 * k2_omega2);

    double k4_theta1 = dt * (s.omega1 + k3_omega1);
    double k4_theta2 = dt * (s.omega2 + k3_omega2);
    double k4_omega1 = dt * f1(s.theta1 + k3_theta1, s.theta2 + k3_theta2, s.omega1 + k3_omega1, s.omega2 + k3_omega2);
    double k4_omega2 = dt * f2(s.theta1 + k3_theta1, s.theta2 + k3_theta2, s.omega1 + k3_omega1, s.omega2 + k3_omega2);

    s.theta1 += (k1_theta1 + 2*k2_theta1 + 2*k3_theta1 + k4_theta1) / 6;
    s.theta2 += (k1_theta2 + 2*k2_theta2 + 2*k3_theta2 + k4_theta2) / 6;
    s.omega1 += (k1_omega1 + 2*k2_omega1 + 2*k3_omega1 + k4_omega1) / 6;
    s.omega2 += (k1_omega2 + 2*k2_omega2 + 2*k3_omega2 + k4_omega2) / 6;
}

void implicit_trapezoidal_method(State &s) {
    State s_new = s;
    double tol = 1e-6;
    double error = 1.0;
    int max_iter = 100;
    int iter = 0;

    while (error > tol && iter < max_iter) {
        double theta1_new = s.theta1 + 0.5 * dt * (s.omega1 + s_new.omega1);
        double theta2_new = s.theta2 + 0.5 * dt * (s.omega2 + s_new.omega2);
        double omega1_new = s.omega1 + 0.5 * dt * (f1(s.theta1, s.theta2, s.omega1, s.omega2) + f1(theta1_new, theta2_new, s_new.omega1, s_new.omega2));
        double omega2_new = s.omega2 + 0.5 * dt * (f2(s.theta1, s.theta2, s.omega1, s.omega2) + f2(theta1_new, theta2_new, s_new.omega1, s_new.omega2));

        error = abs(theta1_new - s_new.theta1) + abs(theta2_new - s_new.theta2) + abs(omega1_new - s_new.omega1) + abs(omega2_new - s_new.omega2);
        s_new.theta1 = theta1_new;
        s_new.theta2 = theta2_new;
        s_new.omega1 = omega1_new;
        s_new.omega2 = omega2_new;
        iter++;
    }

    s = s_new;
}

void predictor_corrector_method(State &s) {
    State s_pred = s; 
    runge_kutta4(s_pred); 
    s = s_pred; 
    implicit_trapezoidal_method(s); 
}


void display(SDL_Window* window, int choise) {
    glClear(GL_COLOR_BUFFER_BIT);

    double x1 = width / 2 + l1 * sin(s.theta1);
    double y1 = height / 4 + l1 * cos(s.theta1);
    double x2 = x1 + l2 * sin(s.theta2);
    double y2 = y1 + l2 * cos(s.theta2);
    
    trajectory.push_back({x2, y2});
    switch (choise){
    case 1:
        runge_kutta4(s);
        break;
    case 2:
        implicit_trapezoidal_method(s);
        break;
    case 3:
        predictor_corrector_method(s);
        break;
    default:
        break;
    }

    glColor3f(0.0, 0.0, 1.0);  
    glBegin(GL_LINE_STRIP);  
    for (const auto& point : trajectory) {
        glVertex2f(point.first, point.second);
    }
    glEnd();

    glColor3f(1.0, 1.0, 1.0);
    glBegin(GL_LINES);
    glVertex2f(width / 2, height / 4);
    glVertex2f(x1, y1);
    glVertex2f(x1, y1);
    glVertex2f(x2, y2);
    glEnd();

    glColor3f(1.0, 0.0, 0.0);
    glPointSize(8.0);
    glBegin(GL_POINTS);
    glVertex2f(x1, y1);
    glVertex2f(x2, y2);
    glEnd();
    
    SDL_GL_SwapWindow(window);
}

int main() {
    int choise;
    cin >> choise;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("Double Pendulum", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_OPENGL);
    SDL_GLContext context = SDL_GL_CreateContext(window);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, height, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);

    bool running = true;
    SDL_Event event;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }
        display(window, choise);
        SDL_Delay(1);
    }
    
    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}
