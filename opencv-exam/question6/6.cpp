#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // 1. 读取图像
    string imgPath = "/home/azure/opencv_exam/question6/picture6/image6.png";
    Mat src = imread(imgPath);
    if (src.empty()) {
        cout << "【错误】无法读取图片，请检查路径！" << endl;
        return -1;
    }

    // 2. 灰度化与二值化
    Mat gray, binary;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    // 3. 去噪
    Mat kernel3 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(binary, binary, MORPH_OPEN, kernel3, Point(-1, -1), 2);

    // 4. 距离变换
    Mat dist;
    distanceTransform(binary, dist, DIST_L2, 3);

    // 5. 寻找局部极大值（峰顶）
    Mat kernel5 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    Mat dist_dilated;
    dilate(dist, dist_dilated, kernel5);

    Mat peaks;
    compare(dist, dist_dilated, peaks, CMP_EQ);

    // 将峰顶绘制成独立的白点，同时过滤掉极小的杂点
    Mat sure_fg_tmp = Mat::zeros(dist.size(), CV_8U);
    for (int r = 0; r < dist.rows; r++) {
        for (int c = 0; c < dist.cols; c++) {
            if (peaks.at<uchar>(r, c) == 255 && dist.at<float>(r, c) > 20.0) {
                circle(sure_fg_tmp, Point(c, r), 4, Scalar(255), -1);
            }
        }
    }

    // ==========================================================
    // 6. 【新核心修复】：消除内部多出的双胞/三胞噪点
    // ==========================================================
    // 使用“闭运算”和“开运算”，清除轻微粘连（从 20 变 16）。
    // 闭运算（膨胀+腐蚀）：如果内部有非常贴近的假点，会把它直接融为一个。
    // 开运算（腐蚀+膨胀）：如果有极细小的孤立白点，会直接把它擦除。
    Mat sure_fg;
    morphologyEx(sure_fg_tmp, sure_fg, MORPH_CLOSE, kernel5);
    morphologyEx(sure_fg, sure_fg, MORPH_OPEN, kernel5);

    // ==========================================================
    // 7. 分水岭流程
    // ==========================================================
    Mat sure_bg;
    dilate(binary, sure_bg, kernel3, Point(-1, -1), 3);

    Mat unknown;
    subtract(sure_bg, sure_fg, unknown);

    Mat markers;
    int num_objects = connectedComponents(sure_fg, markers);

    markers = markers + 1;
    markers.setTo(0, unknown);

    watershed(src, markers);

    // 8. 结果输出
    cout << "---------------------------------------" << endl;
    cout << "自动检测到的圆形目标总数为: " << num_objects << " 个" << endl;
    cout << "---------------------------------------" << endl;

    // 9. 可视化
    vector<Vec3b> colors;
    for (int i = 0; i < num_objects + 1; i++) {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    Mat result = Mat::zeros(src.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= num_objects) {
                result.at<Vec3b>(i, j) = colors[index];
            } else if (index == -1) {
                result.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
            }
        }
    }

    imshow("1. 原始图像", src);
    imshow("2. 确定前景种子 (合并后)", sure_fg);
    imshow("3. 分水岭分割标记图", result);

    cout << "请截图保存 3 个窗口的结果，并按任意键退出。" << endl;
    waitKey(0);
    return 0;
}
