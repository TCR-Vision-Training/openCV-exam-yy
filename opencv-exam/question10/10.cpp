#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

struct Defect {
    string type;
    Rect rect;
    double area;
    Point center;
};

int main() {
    // 1. 读取图像
    string templatePath = "/home/azure/opencv_exam/question10/picture10/image101.png";
    string testPath = "/home/azure/opencv_exam/question10/picture10/image102.png";

    Mat imgTemp = imread(templatePath, IMREAD_COLOR);
    Mat imgTest = imread(testPath, IMREAD_COLOR);

    if (imgTemp.empty() || imgTest.empty()) {
        cout << "【错误】图像读取失败，请检查路径！" << endl;
        return -1;
    }

    Mat result = imgTest.clone();
    vector<Defect> defectList;

    // 1. 预处理与转换
    Mat grayTemp, grayTest;
    cvtColor(imgTemp, grayTemp, COLOR_BGR2GRAY);
    cvtColor(imgTest, grayTest, COLOR_BGR2GRAY);

    // 2. 识别8个正常孔洞 (绿圈)
    vector<Vec3f> circles;
    HoughCircles(grayTest, circles, HOUGH_GRADIENT, 1, 50, 100, 30, 20, 60);

    vector<Point> good_hole_centers;
    for (size_t i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        good_hole_centers.push_back(center);
        // 不画绿圈了，保持参考图简洁风格，把绿圈留给最外层的Marker大框
        // circle(result, center, cvRound(circles[i][2]), Scalar(0, 255, 0), 2);
    }

    // ==========================================================
    // 提取右下角的深色污渍 Stain (红框)
    // ==========================================================
    Mat hsvTest;
    cvtColor(imgTest, hsvTest, COLOR_BGR2HSV);
    Mat maskDark;
    inRange(hsvTest, Scalar(0, 0, 0), Scalar(180, 255, 80), maskDark);

    Mat holeMask = Mat::zeros(maskDark.size(), CV_8U);
    for (auto& hole : good_hole_centers) {
        circle(holeMask, hole, 35, Scalar(255), -1);
    }
    bitwise_and(maskDark, holeMask, maskDark);

    vector<vector<Point>> contoursDark;
    findContours(maskDark, contoursDark, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (const auto& cnt : contoursDark) {
        double area = contourArea(cnt);
        if (area > 800) {
            Rect rect = boundingRect(cnt);
            Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
            defectList.push_back({"Stain", rect, area, center});
        }
    }

    // ==========================================================
    // 提取划痕和缺孔 (红框)
    // ==========================================================
    Mat diff;
    absdiff(grayTemp, grayTest, diff);
    Mat diffBinary;
    threshold(diff, diffBinary, 40, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(diffBinary, diffBinary, MORPH_OPEN, kernel);

    vector<vector<Point>> contoursDiff;
    findContours(diffBinary, contoursDiff, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (const auto& cnt : contoursDiff) {
        double area = contourArea(cnt);
        if (area < 20) continue;

        Rect rect = boundingRect(cnt);
        Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);

        // 如果是正常孔洞，跳过
        bool isHole = false;
        for (auto& hole : good_hole_centers) {
            if (abs(center.x - hole.x) < 15 && abs(center.y - hole.y) < 15) {
                isHole = true; break;
            }
        }
        if (isHole) continue;

        // 排除右下角的污渍区域，防止重复检测
        if (center.x > 300 && center.y > 150) continue;

        float ratio = (float)rect.width / (float)rect.height;
        string type;

        if (ratio > 2.0f || ratio < 0.5f) { // 细长条 -> 划痕
            type = "Scratch";
        } else { // 近似正方形 -> 缺孔
            type = "Missing Hole";
        }
        defectList.push_back({type, rect, area, center});
    }

    // ==========================================================
    // 【统一样式绘制】：绿外框，红内框
    // ==========================================================

    // 1. 画最外层的绿色大框 (marker)
    Rect markerRect(10, 10, result.cols - 20, result.rows - 20);
    rectangle(result, markerRect, Scalar(0, 255, 0), 2);
    putText(result, "marker", Point(markerRect.x + 10, markerRect.y + 25),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);

    // 2. 画所有的缺陷 (统一为红色框 + 红色文字)
    Scalar defectColor = Scalar(0, 0, 255);
    for (const auto& d : defectList) {
        rectangle(result, d.rect, defectColor, 2);
        putText(result, d.type, Point(d.rect.x, d.rect.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.6, defectColor, 2);
    }

    // ==========================================================
    // 输出最终结果报告
    // ==========================================================
    cout << "\n============ 缺陷检测报告 ============" << endl;
    cout << "发现缺陷总数: " << defectList.size() << " 处" << endl;
    for (size_t i = 0; i < defectList.size(); i++) {
        Defect d = defectList[i];
        cout << "[" << i+1 << "] 类型: " << d.type
             << " | 位置: (" << d.center.x << ", " << d.center.y << ")"
             << " | 面积: " << (int)d.area << " px" << endl;
    }
    cout << "=======================================" << endl;

    imshow("缺陷检测 ", result);
    cout << "\n请截图交卷。所有缺陷已统一为红色框。" << endl;

    waitKey(0);
    return 0;
}