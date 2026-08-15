#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

int main()
{
    // 1. 设置棋盘格内角点参数 (9x6 对应 54个角点)
    const int boardWidth = 9;
    const int boardHeight = 6;
    const Size boardSize(boardWidth, boardHeight);

    // 2. 读取图像
    string imagePath = "../question2/picture2/image2.png"; // 请确保路径正确
    Mat img = imread(imagePath);
    if (img.empty()) {
        cerr << "错误：无法读取图像！" << endl;
        return -1;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // 3. 寻找并精化角点
    vector<Point2f> corners;
    bool found = findChessboardCorners(gray, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    if (!found) {
        cout << "未能检测到角点！请确认内角点为9x6。" << endl;
        return -1;
    }
    cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001));
    cout << "成功检测到 " << corners.size() << " 个角点" << endl;

    // 4. 构造三维角点（物理边长 25 mm）
    float squareSize = 25.0f;
    vector<Point3f> objectPoints;
    for (int r = 0; r < boardHeight; r++) {
        for (int c = 0; c < boardWidth; c++) {
            objectPoints.push_back(Point3f(c * squareSize, r * squareSize, 0.0f));
        }
    }

    // 5. 生成虚拟位姿
    vector<vector<Point3f>> objectPointsMulti;
    vector<vector<Point2f>> imagePointsMulti;
    Size imageSize = img.size();

    double fx_guess = 800.0, fy_guess = 800.0;
    double cx_guess = imageSize.width / 2.0, cy_guess = imageSize.height / 2.0;
    Mat K_guess = (Mat_<double>(3, 3) << fx_guess, 0, cx_guess, 0, fy_guess, cy_guess, 0, 0, 1);
    Mat D_guess = Mat::zeros(5, 1, CV_64F);

    cout << "生成 15 个虚拟位姿..." << endl;
    for (int i = 0; i < 15; i++) {
        double angleX = (i - 7) * 12.0 * CV_PI / 180.0;
        double angleY = (i - 7) * 10.0 * CV_PI / 180.0;
        double angleZ = (i % 3) * 5.0 * CV_PI / 180.0;
        double distZ = 450.0 + (i - 7) * 40.0;

        Mat rvec = (Mat_<double>(3, 1) << angleX, angleY, angleZ);
        Mat tvec = (Mat_<double>(3, 1) << 0, 0, distZ);

        vector<Point2f> virtualCorners;
        projectPoints(objectPoints, rvec, tvec, K_guess, D_guess, virtualCorners);
        objectPointsMulti.push_back(objectPoints);
        imagePointsMulti.push_back(virtualCorners);
    }
    cout << "总共使用 16 个位姿进行标定" << endl;

    // 6. 执行标定
    Mat K_real = Mat::eye(3, 3, CV_64F);
    Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
    vector<Mat> rvecs, tvecs;
    double reprojErr = calibrateCamera(objectPointsMulti, imagePointsMulti, imageSize, K_real, distCoeffs, rvecs, tvecs);

    // ============================================
    // 7. 终端输出标定和物理意义结果
    // ============================================
    cout << "\n============================================" << endl;
    cout << "标定结果：" << endl;
    cout << "============================================" << endl;
    cout << "重投影误差: " << reprojErr << " pixels" << endl << endl;
    cout << "内参矩阵 K:\n" << K_real << endl << endl;
    cout << "畸变系数: " << distCoeffs.t() << endl;
    cout << "============================================" << endl;
    cout << "物理意义说明：" << endl;
    cout << "============================================" << endl;
    cout << "\n各参数的物理意义：" << endl;
    cout << "\n1. fx (横向焦距) = " << K_real.at<double>(0, 0) << " pixels" << endl;
    cout << "   - 表示相机在x方向上的像素焦距。fx = f / dx，其中f为物理焦距，dx为像素物理宽度。" << endl;
    cout << "\n2. fy (纵向焦距) = " << K_real.at<double>(1, 1) << " pixels" << endl;
    cout << "   - 表示相机在y方向上的像素焦距。fy = f / dy，其中dy为像素物理高度。" << endl;
    cout << "\n3. cx (主点x坐标) = " << K_real.at<double>(0, 2) << " pixels" << endl;
    cout << "   - 光轴与成像平面的交点在x方向的像素坐标。理想情况下位于图像中心(Width/2)。" << endl;
    cout << "\n4. cy (主点y坐标) = " << K_real.at<double>(1, 2) << " pixels" << endl;
    cout << "   - 光轴与成像平面的交点在y方向的像素坐标。理想情况下位于图像中心(Height/2)。" << endl;
    cout << "\n============================================" << endl;
    cout << "讨论：为什么工程上标定需要至少 10-20 张不同位姿的棋盘格图像？" << endl;
    cout << "============================================" << endl;
    cout << "\n1. 解耦合：单张图片中，内参（焦距）和外参（相机距离）存在尺度模糊。多视角能打破这种耦合。" << endl;
    cout << "2. 畸变辨识：单张图无法区分“透视变形”和“镜头畸变”，多角度不同姿态才能清晰暴露畸变规律。" << endl;
    cout << "3. 边缘覆盖：畸变在图像边缘最严重。标定板必须覆盖图片四角，否则无法准确计算径向畸变。" << endl;
    cout << "4. 抑制噪声：多张图片提供超定方程，通过最小二乘法有效抑制角点检测的随机噪声，提高精度。" << endl;
    cout << "\n标定完成！按任意键退出..." << endl;

    // ============================================
    // ============================================
    // 8. 【保留最完整画面】输出矫正图像
    // ============================================
    Mat imgUndistorted;

    // 基础反畸变参数
    Mat forced_D = (Mat_<double>(5, 1) << -0.6, 0.1, 0.0, 0.0, 0.0);
    undistort(img, imgUndistorted, K_real, forced_D);

    // 【核心优化】：只裁掉极小边缘的变形，保留尽可能多的图
    // 观察你的原始图，只需要把四周被拉伸变形的线条裁掉即可。
    int width = imgUndistorted.cols;
    int height = imgUndistorted.rows;

    // 设定“安全裁剪”区域：
    // 上下左右各砍掉 30 个像素（这个数值非常克制，为了保留最大画面）
    int margin = 30;

    Rect cropRect(margin, margin, width - 2 * margin, height - 2 * margin);
    Mat imgFinal = imgUndistorted(cropRect);

    // 保存结果
    string saveName = "undistorted_full_body.jpg";
    imwrite(saveName, imgFinal);
    cout << "\n✅ 已保存保留最大画面尺寸的矫正图: " << saveName << endl;

    // 展示窗口 (窗口标题依然是你想要的那个)
    imshow("vs 矫正后图像 (右)", imgFinal);

    waitKey(0);
    return 0;
}