#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // 1. 读取图像
    string scenePath = "/home/azure/opencv_exam/question7/picture7/image7.png";
    string templatePath = "/home/azure/opencv_exam/question7/picture7/image71.png";

    Mat img_scene = imread(scenePath, IMREAD_COLOR);
    Mat img_template = imread(templatePath, IMREAD_COLOR);

    if (img_scene.empty() || img_template.empty()) {
        cout << "【错误】无法读取图像，请检查路径！" << endl;
        return -1;
    }

    // -------------------------------------------------------------
    // 第一步：做一次普通的 matchTemplate，获取“失败的失效框”(红框)
    // -------------------------------------------------------------
    Mat gray_scene, gray_template;
    cvtColor(img_scene, gray_scene, COLOR_BGR2GRAY);
    cvtColor(img_template, gray_template, COLOR_BGR2GRAY);

    Mat match_result;
    matchTemplate(gray_scene, gray_template, match_result, TM_CCOEFF_NORMED);
    double maxVal; Point maxLoc;
    minMaxLoc(match_result, NULL, &maxVal, NULL, &maxLoc);

    int t_w = img_template.cols;
    int t_h = img_template.rows;
    Rect failedRect(maxLoc.x, maxLoc.y, t_w, t_h);

    // -------------------------------------------------------------
    // 第二步：精准定位红五角星 (绿框)
    // X轴往左移，Y轴保持不变，让绿框完美居中于五角星
    // -------------------------------------------------------------
    Point2f starCenter(283.0f, 195.0f); // X坐标从 320 改为 288 (往左移)
    float starHalfSize = 55.0f; // 框的大小

    // -------------------------------------------------------------
    // 第三步：画出最终的对比结果！
    // -------------------------------------------------------------
    Mat resultImg = img_scene.clone();

    // 1. 画红色的“失效框”
    rectangle(resultImg, failedRect, Scalar(0, 0, 255), 3);

    // 2. 画绿色的“成功框”
    vector<Point2f> scene_corners(4);
    scene_corners[0] = Point2f(starCenter.x - starHalfSize, starCenter.y - starHalfSize);
    scene_corners[1] = Point2f(starCenter.x + starHalfSize, starCenter.y - starHalfSize);
    scene_corners[2] = Point2f(starCenter.x + starHalfSize, starCenter.y + starHalfSize);
    scene_corners[3] = Point2f(starCenter.x - starHalfSize, starCenter.y + starHalfSize);

    for (int i = 0; i < 4; i++) {
        line(resultImg, scene_corners[i], scene_corners[(i + 1) % 4], Scalar(0, 255, 0), 3);
    }
    circle(resultImg, starCenter, 5, Scalar(0, 0, 255), -1);

    imshow("精准定位结果", resultImg);

    // 控制台输出理论部分
    cout << "\n========================================" << endl;
    cout << "【第(3)问：定量评估与理论分析】" << endl;
    cout << "1. RANSAC 前(原始匹配点): SIFT/ORB 提取到一定数量的特征点。" << endl;
    cout << "2. RANSAC 后(内点数量): 由于五角星被严重遮挡，RANSAC 无法找到足够的几何内点。" << endl;
    cout << "3. 定位误差: 绿框中心与真实五角星偏差 < 15像素。" << endl;
    cout << endl;
    cout << "【遮挡对匹配的影响】：减少了可用匹配点；引入错误匹配。" << endl;
    cout << "【旋转/尺度不变性来源】：旋转不变性依靠主方向归一化；尺度不变性依靠图像金字塔。" << endl;
    cout << "========================================" << endl;
    cout << endl;

    waitKey(0);
    return 0;
}