#include "person.h"
#include "opencv2/highgui/highgui.hpp"

Person::Person(yolov7_ros::Object Operson){
    _npCenter.x = Operson.center.x; //col
    _npCenter.y = Operson.center.y; //row
    _FlowSum.x = 0;
    _FlowSum.y = 0;
    for(int i=0;i<4;i++){
        _npCorner[i].x = Operson.bounding_box_2d.corners[i].kp[0];
        _npCorner[i].y = Operson.bounding_box_2d.corners[i].kp[1];
    }

    _dHeight = _npCorner[1].y-_npCorner[0].y;
    _dWidth = _npCorner[2].x-_npCorner[1].x;

}
void Person::CalculateAVG(cv::Mat img){
    double dSize = _dHeight*_dWidth;
    _nSize = dSize;
    double dAVG = 0;
    for(int i = _npCorner[0].x ; i <= _npCorner[3].x ; i++){ //col
        for(int j = _npCorner[0].y ; j <= _npCorner[1].y ; j++){ //row
            dAVG += ((double)((int)img.at<uchar>(j,i)))/(std::pow(0.2,dSize)+dSize);
        }
    }
    dAVG /= 2;
    _dLocalAVG = dAVG;
}
bool Person::InPerson(cv::Point2f point){
    if(point.y < _npCorner[0].y || point.y > _npCorner[1].y)return false;
    if(point.x < _npCorner[0].x || point.x > _npCorner[3].x)return false;
    return true;
}
void Person::FlowSuming(cv::Point2f point){
    _FlowSum += point;
    _nFeatNum++;
}
void Person::Calculate(void){
    _FlowAVG = _FlowSum/((double)_nFeatNum);
    _dFlowMag = std::sqrt(std::pow(_FlowAVG.x,2)+std::pow(_FlowAVG.y,2));

    //
    // if(_dFlowMag > 1.4*_dBGflowMag || _dFlowMag < 0.6*_dBGflowMag)_bDynamic = true;
    
    // double dCosine;
    // dCosine = _BackGroundFlow.dot(_FlowAVG)/(_dBGflowMag*_dFlowMag);
    // if(dCosine < 0.3)_bDynamic = true;

    //
    cv::Point2f error = _BackGroundFlow-_FlowAVG;
    double dErrorMag = std::sqrt(std::pow(error.x,2)+std::pow(error.y,2));
    //std::cout<< _dBGflowMag << " // "<<dErrorMag<<"\n";

    double cl = 0.15*std::sqrt(std::sqrt(std::pow(_npCenter.x-320,2) + std::pow(_npCenter.y-240,2)));
    if(dErrorMag > 0.25*_dBGflowMag*cl)_bDynamic = true;

    if(_nFeatNum < 6)_bDynamic = false;
}
void Person::SetBackgroundFlow(cv::Point2f point, double magnitude){
    _BackGroundFlow = point;
    _dBGflowMag = magnitude;
}
double Person::MovingAVG(void){
    return _dLocalAVG;
}
cv::Point2d Person::Center(void){
    return _npCenter;
}
int Person::Size(void){
    return _nSize;
}
int Person::FeatureNum(void){
    return _nFeatNum;
}
cv::Point2f Person::FlowSumVector(void){
    return _FlowSum;
}
cv::Point2f Person::FlowAVGvector(void){
    return _FlowAVG;
}
double Person::FlowMagnitude(void){
    return _dFlowMag;
}
bool Person::IsDynamic(void){
    return _bDynamic;
}