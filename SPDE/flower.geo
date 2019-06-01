R = 2.0;
N = 10;
Rin = 0.5;

Mesh.CharacteristicLengthFactor = 0.5;
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
//Mesh.CharacteristicLengthExtendFromBoundary = 1;

Mesh.CharacteristicLengthMin = 0.01;
Mesh.CharacteristicLengthMax = 0.1;

Point(1) = {0.0, 0.0, 0};
Point(2) = {0.0, R*Cos(Pi/N), 0};
Point(3) = {R*Sin(Pi/N), R*Cos(Pi/N), 0};

Point(4) = {0.1, 1.9, 0};
Point(5) = {0.26, 1.86, 0};
Point(6) = {0.38, 1.81, 0};
Point(7) = {0.45, 1.73, 0};
Point(8) = {0.5, 1.64, 0};
Point(9) = {0.49, 1.54, 0};
Point(10) = {0.45, 1.46, 0};
Point(11) = {0.37, 1.39, 0};
Point(12) = {0.31, 1.31, 0};
Point(13) = {0.27, 1.25, 0};
Point(14) = {0.26, 1.16, 0};
Point(15) = {0.26, 1.09, 0};
Point(16) = {0.27, 1.05, 0};
Point(17) = {1.07*Sin(Pi/N), 1.07*Cos(Pi/N), 0};

BSpline(4) = {2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};

Point(100) = { Rin*Sin(Pi/N), Rin*Cos(Pi/N), 0 };
Point(101) = { 0.0, Rin, 0 };

Circle(5) = {100, 1, 101};

Symmetry {1, 0, 0, 0} { Duplicata { Curve{4}; Curve{5}; } }

For k In {1:N:1}
    Rotate {{0, 0, 1}, {0, 0, 0}, 2*k*Pi/N} { Duplicata { Curve{4,5,6,7}; } }
EndFor

Curve Loop(1) = {4, -42, 40, -38, 36, -34, 32, -30, 28, -26, 24, -22, 20, -18, 16, -14, 12, -10, 8, -6};
Curve Loop(2) = {7, -5, 43, -41, 39, -37, 35, -33, 31, -29, 27, -25, 23, -21, 19, -17, 15, -13, 11, -9};
Plane Surface(1) = {1, 2};

Physical Surface("flower") = {1};
Physical Curve("outer") = {8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 4, 6};
Physical Curve("inner") = {7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 5};


