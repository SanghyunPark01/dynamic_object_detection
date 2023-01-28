#ifndef PERSON_H
#define PERSON_H

#include <yolov7_ros/ObjectsStamped.h>
#include "opencv2/highgui/highgui.hpp"

class Person{
private:
    cv::Point2d _npCenter;
    cv::Point2d _npCorner[4];//0 : LU // 1 : LD // 2 : RD // 3 : RU
    double _dLocalAVG;
    double _dHeight;
    double _dWidth;
    int _nSize;
    cv::Point2f _FlowSum;
    int _nFeatNum = 0;
    cv::Point2f _FlowAVG;
    double _dFlowMag;
    cv::Point2f _BackGroundFlow;
    double _dBGflowMag;
    bool _bDynamic = false;
public:
    Person(yolov7_ros::Object Operson);
    void CalculateAVG(cv::Mat img);
    double MovingAVG(void);
    cv::Point2d Center(void);
    int Size(void);
    bool InPerson(cv::Point2f point);
    void FlowSuming(cv::Point2f point);
    void Calculate(void);
    void SetBackgroundFlow(cv::Point2f point, double magnitude);
    int FeatureNum(void);
    cv::Point2f FlowSumVector(void);
    cv::Point2f FlowAVGvector(void);
    double FlowMagnitude(void);
    bool IsDynamic(void);
};


#endif