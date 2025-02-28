#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <GL/gl.h>
#include <GL/glu.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

using namespace std;

const double g = 9.81;
const double m1 = 1.0, m2 = 3.0;
const double l1 = 100.0, l2 = 100.0;
const double dt = 0.03;
const int width = 800, height = 600;

struct State {
    double theta1, theta2, omega1, omega2;
};

State s = {M_PI / 2, M_PI / 2, 0.0, 0.0};

//State s = {0.2, -0.1, 0.0, 0.0}; // не переворачивается



vector<pair<double, double>> trajectory; 
vector<double> time_steps;  

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



void dormand_prince_8(State &s) {
    // Коэффициенты таблицы Бутчера
    const double a[13][13] = {
        {0},
        { 1.0 / 18},
        {1.0 / 48, 1.0 / 16},
        { 1.0 / 32, 0, 3.0 / 32},
        { 5.0 / 16, 0, -75.0 / 64, 75.0 / 64},
        { 3.0 / 80, 0, 0, 3.0 / 16, 3.0 / 20},
        { 29443841.0 / 614563906, 0, 0, 77736538.0 / 692538347, -28693883.0 / 1125000000, 23124283.0 / 1800000000},
        { 16016141.0 / 946692911, 0, 0, 61564180.0 / 158732637, 22789713.0 / 633445777, 545815736.0 / 2771057229, -180193667.0 / 1043307555},
        { 39632708.0 / 573591083, 0, 0, -433636366.0 / 683701615, -421739975.0 / 2616292301, 100302831.0 / 723423059, 790204164.0 / 839813087, 800635310.0 / 3783071287},
        { 246121993.0 / 1340847787, 0, 0, -37695042795.0 / 15268766246, -309121744.0 / 1061227803, -12992083.0 / 490766935, 6005943493.0 / 2108947869, 393006217.0 / 1396673457, 123872331.0 / 1001029789},
        { -1028468189.0 / 846180014, 0, 0, 8478235783.0 / 508512852, 1311729495.0 / 1432422823, -10304129995.0 / 1701304382, -48777925059.0 / 3047939560, 15336726248.0 / 1032824649, -45442868181.0 / 3398467696, 3065993473.0 / 597172653},
        { 185892177.0 / 718116043, 0, 0, -3185094517.0 / 667107341, -477755414.0 / 1098053517, -703635378.0 / 230739211, 5731566787.0 / 1027545527, 5232866602.0 / 850066563, -4093664535.0 / 808688257, 3962137247.0 / 1805957418, 65686358.0 / 487910083},
        { 403863854.0 / 491063109, 0, 0, -5068492393.0 / 434740067, -411421997.0 / 543043805, 652783627.0 / 914296604, 11173962825.0 / 925320556, -13158990841.0 / 6184727034, 3936647629.0 / 1978049680, -160528059.0 / 685178525, 248638103.0 / 1413531060, 0}
    };

    const double b[13] = {
        14005451.0 / 335480064, 0, 0, 0, 0, -59238493.0 / 1068277825, 181606767.0 / 758867731, 561292985.0 / 797845732, -1041891430.0 / 1371343529, 760417239.0 / 1151165299, 118820643.0 / 751138087, -528747749.0 / 2220607170, 1.0 / 4
    };

    vector<double> k1_theta1(13), k1_theta2(13), k1_omega1(13), k1_omega2(13);


    for (int i = 0; i < 13; ++i) {
        double sum_theta1 = 0, sum_theta2 = 0, sum_omega1 = 0, sum_omega2 = 0;
        for (int j = 0; j < i; ++j) {
            sum_theta1 += a[i][j] * k1_theta1[j];
            sum_theta2 += a[i][j] * k1_theta2[j];
            sum_omega1 += a[i][j] * k1_omega1[j];
            sum_omega2 += a[i][j] * k1_omega2[j];
        }
        k1_theta1[i] = dt * (s.omega1 + sum_omega1);
        k1_theta2[i] = dt * (s.omega2 + sum_omega2);
        k1_omega1[i] = dt * f1(s.theta1 + sum_theta1, s.theta2 + sum_theta2, s.omega1 + sum_omega1, s.omega2 + sum_omega2);
        k1_omega2[i] = dt * f2(s.theta1 + sum_theta1, s.theta2 + sum_theta2, s.omega1 + sum_omega1, s.omega2 + sum_omega2);
    }

    double delta_theta1 = 0, delta_theta2 = 0, delta_omega1 = 0, delta_omega2 = 0;
    for (int i = 0; i < 13; ++i) {
        delta_theta1 += b[i] * k1_theta1[i];
        delta_theta2 += b[i] * k1_theta2[i];
        delta_omega1 += b[i] * k1_omega1[i];
        delta_omega2 += b[i] * k1_omega2[i];
    }

    s.theta1 += delta_theta1;
    s.theta2 += delta_theta2;
    s.omega1 += delta_omega1;
    s.omega2 += delta_omega2;
}

void crank_nicolson(State &s) {
    double theta1_new = s.theta1;
    double theta2_new = s.theta2;
    double omega1_new = s.omega1;
    double omega2_new = s.omega2;

    for (int iter = 0; iter < 10; ++iter) {
        double theta1_mid = (s.theta1 + theta1_new) / 2;
        double theta2_mid = (s.theta2 + theta2_new) / 2;
        double omega1_mid = (s.omega1 + omega1_new) / 2;
        double omega2_mid = (s.omega2 + omega2_new) / 2;

        double delta_theta1 = theta1_new - s.theta1 - dt / 2 * (s.omega1 + omega1_new);
        double delta_theta2 = theta2_new - s.theta2 - dt / 2 * (s.omega2 + omega2_new);
        double delta_omega1 = omega1_new - s.omega1 - dt / 2 * (f1(s.theta1, s.theta2, s.omega1, s.omega2) + f1(theta1_new, theta2_new, omega1_new, omega2_new));
        double delta_omega2 = omega2_new - s.omega2 - dt / 2 * (f2(s.theta1, s.theta2, s.omega1, s.omega2) + f2(theta1_new, theta2_new, omega1_new, omega2_new));

        if (abs(delta_theta1) + abs(delta_theta2) + abs(delta_omega1) + abs(delta_omega2) < 1e-6) {
            break;
        }

        theta1_new -= delta_theta1;
        theta2_new -= delta_theta2;
        omega1_new -= delta_omega1;
        omega2_new -= delta_omega2;
    }

    s.theta1 = theta1_new;
    s.theta2 = theta2_new;
    s.omega1 = omega1_new;
    s.omega2 = omega2_new;
}

void bogacki_shampine(State &s) {
    double t = dt;
    double k1_theta1, k1_theta2, k1_omega1, k1_omega2;
    double k2_theta1, k2_theta2, k2_omega1, k2_omega2;
    double k3_theta1, k3_theta2, k3_omega1, k3_omega2;
    double k4_theta1, k4_theta2, k4_omega1, k4_omega2;


    k1_theta1 = s.omega1;
    k1_theta2 = s.omega2;
    k1_omega1 = f1(s.theta1, s.theta2, s.omega1, s.omega2);
    k1_omega2 = f2(s.theta1, s.theta2, s.omega1, s.omega2);

 
    k2_theta1 = s.omega1 + 0.5 * t * k1_omega1;
    k2_theta2 = s.omega2 + 0.5 * t * k1_omega2;
    k2_omega1 = f1(s.theta1 + 0.5 * t * k1_theta1, s.theta2 + 0.5 * t * k1_theta2, s.omega1 + 0.5 * t * k1_omega1, s.omega2 + 0.5 * t * k1_omega2);
    k2_omega2 = f2(s.theta1 + 0.5 * t * k1_theta1, s.theta2 + 0.5 * t * k1_theta2, s.omega1 + 0.5 * t * k1_omega1, s.omega2 + 0.5 * t * k1_omega2);


    k3_theta1 = s.omega1 + 0.75 * dt * k2_omega1;
    k3_theta2 = s.omega2 + 0.75 * dt * k2_omega2;
    k3_omega1 = f1(s.theta1 + 0.75 * t * k2_theta1, s.theta2 + 0.75 * t * k2_theta2, s.omega1 + 0.75 * t * k2_omega1, s.omega2 + 0.75 * t * k2_omega2);
    k3_omega2 = f2(s.theta1 + 0.75 * t * k2_theta1, s.theta2 + 0.75 * t * k2_theta2, s.omega1 + 0.75 * t * k2_omega1, s.omega2 + 0.75 * t * k2_omega2);

  
    State y_new;
    y_new.theta1 = s.theta1 + t * (2.0 / 9.0 * k1_theta1 + 1.0 / 3.0 * k2_theta1 + 4.0 / 9.0 * k3_theta1);
    y_new.theta2 = s.theta2 + t * (2.0 / 9.0 * k1_theta2 + 1.0 / 3.0 * k2_theta2 + 4.0 / 9.0 * k3_theta2);
    y_new.omega1 = s.omega1 + t * (2.0 / 9.0 * k1_omega1 + 1.0 / 3.0 * k2_omega1 + 4.0 / 9.0 * k3_omega1);
    y_new.omega2 = s.omega2 + t * (2.0 / 9.0 * k1_omega2 + 1.0 / 3.0 * k2_omega2 + 4.0 / 9.0 * k3_omega2);


    k4_theta1 = y_new.omega1;
    k4_theta2 = y_new.omega2;
    k4_omega1 = f1(y_new.theta1, y_new.theta2, y_new.omega1, y_new.omega2);
    k4_omega2 = f2(y_new.theta1, y_new.theta2, y_new.omega1, y_new.omega2);

    State z_new;
    z_new.theta1 = s.theta1 + t * (7.0 / 24.0 * k1_theta1 + 1.0 / 4.0 * k2_theta1 + 1.0 / 3.0 * k3_theta1 + 1.0 / 8.0 * k4_theta1);
    z_new.theta2 = s.theta2 + t * (7.0 / 24.0 * k1_theta2 + 1.0 / 4.0 * k2_theta2 + 1.0 / 3.0 * k3_theta2 + 1.0 / 8.0 * k4_theta2);
    z_new.omega1 = s.omega1 + t * (7.0 / 24.0 * k1_omega1 + 1.0 / 4.0 * k2_omega1 + 1.0 / 3.0 * k3_omega1 + 1.0 / 8.0 * k4_omega1);
    z_new.omega2 = s.omega2 + t * (7.0 / 24.0 * k1_omega2 + 1.0 / 4.0 * k2_omega2 + 1.0 / 3.0 * k3_omega2 + 1.0 / 8.0 * k4_omega2);

    double error_theta1 = abs(y_new.theta1 - z_new.theta1);
    double error_theta2 = abs(y_new.theta2 - z_new.theta2);
    double error_omega1 = abs(y_new.omega1 - z_new.omega1);
    double error_omega2 = abs(y_new.omega2 - z_new.omega2);
    double max_error = max(max(error_theta1, error_theta2), max(error_omega1, error_omega2));
    double step = 0.0;

    if (max_error > 0) {
        double scale = pow(1e-6 / max_error, 1.0 / 3.0);
        t *= min(scale, 2.0); 
    }
    time_steps.push_back(t);
    s = y_new;
}

void predictor_corrector_method(State &s) {
    State s_pred = s; 
    runge_kutta4(s_pred); 
    s = s_pred; 
    crank_nicolson(s); 
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
        crank_nicolson(s);
        break;
    case 3:
        predictor_corrector_method(s);
        break;
    case 4:
        dormand_prince_8(s);
        break;
    case 5:
        bogacki_shampine(s);
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

    for (const auto& step : time_steps) {
        cout << "t = " << step<< endl;
    }
    
    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}