// Markus Buchholz, 2023
// g++ torque_control.cpp -o t -I/usr/include/python3.8 -lpython3.8

#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <tuple>
#include <cmath>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

float L1 = 2.0;
float L2 = 2.0;
float g = 9.81;
float m1 = 1.0;
float m2 = 1.0;
 float dt = 0.001;

float kp = 0.5;
float kv = 1.0;

//--------------------------------------------------------------------------------------------
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>> IK(std::vector<float> X, std::vector<float> Y)
{

    std::vector<float> theta1u;
    std::vector<float> theta2u;

    std::vector<float> theta1d;
    std::vector<float> theta2d;

    for (int ii = 0; ii < X.size(); ii++)
    {

        float r = std::sqrt((X[ii] * X[ii] + Y[ii] * Y[ii]));

        float c = (r * r - L1 * L1 + L2 * L2) / (2 * L1 * L2);

        float d1 = std::sqrt(1 - c * c);
        float d2 = -std::sqrt(1 - c * c);

        theta2u.push_back(std::atan2(d1, c));
        theta2d.push_back(std::atan2(d2, c));

        float tu = std::atan2(Y[ii], X[ii]) - std::atan2(L2 * std::sin(std::atan2(d1, c)), L1 + L2 * std::cos(std::atan2(d1, c)));
        float td = std::atan2(Y[ii], X[ii]) - std::atan2(L2 * std::sin(std::atan2(d2, c)), L1 + L2 * std::cos(std::atan2(d2, c)));

        theta1u.push_back(tu);
        theta1d.push_back(td);
    }

    return std::make_tuple(theta1u, theta2u, theta1d, theta2d);
}
//---------------------------------------------------------------------------------------------------------

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> generatePath()
{

    float tmax = 7.0;
    std::vector<float> X;
    std::vector<float> Y;
    std::vector<float> time;

    for (float t = 0; t < tmax; t = t + dt)
    {

        X.push_back(2.0 + 0.5 * std::sin(3.0 * t));
        Y.push_back(1.0 + 0.5 * std::cos(3.0 * t));
        time.push_back(t);
    }

    return std::make_tuple(X, Y, time);
}
//---------------------------------------------------------------------------------------------------------

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>> computeDesiredTraj(std::vector<float> qd1x, std::vector<float> qd2y)
{

    std::vector<float> qd1;
    std::vector<float> qd2;
    std::vector<float> qdp1;
    std::vector<float> qdp2;
    std::vector<float> qdpp1;
    std::vector<float> qdpp2;
    float A = 0.1;
    float fact = 3.0;
    float t = 0;

    for (int ii = 0; ii < qd1x.size(); ii++)
    {

        qd1.push_back(A * std::sin(fact * t));
        qd2.push_back(A * std::cos(fact * t));

        qdp1.push_back(A * fact * std::cos(fact * t));
        qdp2.push_back(-A * fact * std::sin(fact * t));

        qdpp1.push_back(-A * fact * fact * std::sin(fact * t));
        qdpp2.push_back(-A * fact * fact * std::cos(fact * t));

        t += dt;
    }

    return std::make_tuple(qd1, qd2, qdp1, qdp2, qdpp1, qdpp2);
}

//---------------------------------------------------------------------------------------------------------

std::tuple<float, float, float, float> computeTorquesController(float x1, float x2, float x3, float x4, float qd1, float qd2, float qdp1, float qdp2, float qdpp1, float qdpp2)
{

    float a1 = L1;
    float a2 = L2;

    float e1 = qd1 - x1;
    float e2 = qd2 - x2;
    float ep1 = qdp1 - x3;
    float ep2 = qdp2 - x4;

    float m11 = (m1 + m2) * a1 * a1 + m2 * a2 * a2 + 2 * m2 * a1 * a2 * std::cos(x2);

    float m12 = m2 * a2 * a2 + m2 * a1 * a2 * std::cos(x2);

    float m22 = m2 * a2 * a2;

    float n1 = -m2 * a1 * a2 * (2 * x3 * x4 + x4 * x4) * std::sin(x2);
    n1 = n1 + (m1 + m2) * g * a1 * cos(x1) + m2 * g * a2 * std::cos(x1 + x2);

    float n2 = m2 * a1 * a2 * x3 * x3 * std::sin(x2) + m2 * g * a2 * std::cos(x1 + x2);

    float s1 = qdpp1 + kv * ep1 + kp * e1;
    float s2 = qdpp2 + kv * ep2 + kp * e2;

    // torques
    float t1 = m11 * s1 + m12 * s2 + n1;
    float t2 = m12 * s1 + m22 * s2 + n2;

    return std::make_tuple(t1, t2, e1, e2);
}

//---------------------------------------------------------------------------------------------------------

std::tuple<float, float, float, float> robotArmDynamics(float x1, float x2, float x3, float x4, float t1, float t2)
{

    float a1 = L1;
    float a2 = L2;

    float m11 = (m1 + m2) * a1 * a1 + m2 * a2 * a2 + 2 * m2 * a1 * a2 * std::cos(x2);

    float m12 = m2 * a2 * a2 + m2 * a1 * a2 * std::cos(x2);

    float m22 = m2 * a2 * a2;

    float det = m11 * m22 - m12 * m12;

    float mI11 = m22 / det;
    float mI12 = -m12 / det;
    float mI22 = m11 / det;

    float n1 = -m2 * a1 * a2 * (2 * x3 * x4 + x4 * x4) * std::sin(x2);
    n1 = n1 + (m1 + m2) * g * a1 * cos(x1) + m2 * g * a2 * std::cos(x1 + x2);

    
    float n2 = m2 * a1 * a2 * x3 * x3 * std::sin(x2) + m2 * g * a2 * std::cos(x1 + x2);

    // states
    float xp1 = x3;
    float xp2 = x4;
    float xp3 = mI11 * (-n1 + t1) + mI12 * (-n2 + t2);
    float xp4 = mI12 * (-n1 + t1) + mI22 * (-n2 + t2);

    return std::make_tuple(xp1, xp2, xp3, xp4);
}

//---------------------------------------------------------------------------------------------------------

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>> runSimulation()
{

    auto path = generatePath();
    auto traj = computeDesiredTraj(std::get<0>(path), std::get<1>(path));

    // states
    float x1 = 0;
    float x2 = 0;
    float x3 = 0;
    float x4 = 0;

    std::vector<float> y1;
    std::vector<float> y2;
    
    std::vector<float> t1;
    std::vector<float> t2;

    for (int ii = 0; ii < std::get<0>(traj).size(); ii++)
    {

        auto torques = computeTorquesController(x1, x2, x3, x4, std::get<0>(traj)[ii], std::get<1>(traj)[ii], std::get<2>(traj)[ii], std::get<3>(traj)[ii], std::get<4>(traj)[ii], std::get<5>(traj)[ii]);
        auto states = robotArmDynamics(x1, x2, x3, x4, std::get<0>(torques), std::get<1>(torques));
        x1 = std::get<0>(states);
        x2 = std::get<1>(states);
        x3 = std::get<2>(states);
        x4 = std::get<3>(states);

        y1.push_back(x1);
        y2.push_back(x2);
        t1.push_back(std::get<0>(torques));
        t2.push_back(std::get<1>(torques));
    }

    return std::make_tuple(t1, t2, y1, y2, std::get<2>(path));
}

//---------------------------------------------------------------------------------------------------------

void plot2D(std::vector<float> t1, std::vector<float> t2, std::vector<float> time)
{
    plt::title("Computed robot torques. ");
    plt::named_plot("torque motor 1", time, t1);
    plt::named_plot("torque motor 2", time, t2);
    plt::xlabel("time");
    plt::ylabel("torque");
    plt::legend();

    plt::show();
}

//---------------------------------------------------------------------------------------------------------

void plotPath(std::vector<float> X, std::vector<float> Y)
{
    plt::title("2R robot path. ");
    plt::named_plot("robot path", X, Y);
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::legend();
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::show();
}

//---------------------------------------------------------------------------------------------------------

int main()
{
    auto path = generatePath();
    auto motion1 = IK(std::get<0>(path), std::get<1>(path));
    plot2D(std::get<0>(motion1), std::get<1>(motion1), std::get<2>(path));
    plotPath(std::get<0>(path), std::get<1>(path));
    
    
    auto torques = runSimulation();
    plot2D(std::get<0>(torques), std::get<1>(torques), std::get<4>(torques));

}
