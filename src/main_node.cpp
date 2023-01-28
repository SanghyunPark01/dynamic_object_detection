#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/highgui/highgui.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <std_msgs/Int32.h>
#include <yolov7_ros/ObjectsStamped.h>
#include "person.h"
#include <opencv2/video/tracking.hpp>
cv::Mat cv_mPreImgC;
cv::Mat cv_mPreDepth;
bool bFlag = false;

std::vector<cv::Point2f> p0, p1;
bool bDynamic_flag = false;

void Test(cv::Mat imgP, cv::Mat imgC, std::vector<yolov7_ros::Object> detected_person);
class NodeServer{
private:
    ros::NodeHandle _nh;
    ros::Publisher _pub;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    message_filters::Subscriber<sensor_msgs::Image> _rgb_sub;
    message_filters::Subscriber<sensor_msgs::Image> _depth_sub;
    boost::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> _sync;
    ros::Subscriber _sub_Object;

    std::vector<yolov7_ros::Object> _v_detected_person;
public:
    NodeServer(){
        _pub = _nh.advertise<std_msgs::Int32>("/dynamic_flag",1);

        _sub_Object = _nh.subscribe("/yolov7/yolov7_detection",10, &NodeServer::callback2, this);

        _rgb_sub.subscribe(_nh, "/yolov7/detected_image", 1);
        _depth_sub.subscribe(_nh, "/yolov7/detected_depth", 1);
  
        _sync.reset(new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(1), _rgb_sub, _depth_sub));

        _sync->registerCallback(boost::bind(&NodeServer::callback, this, _1, _2));

    }
    void callback(const sensor_msgs::ImageConstPtr &rgb_msg, const sensor_msgs::ImageConstPtr &depth_msg);
    void callback2(const yolov7_ros::ObjectsStamped& detect_result);
};
void NodeServer::callback(const sensor_msgs::ImageConstPtr &rgb_msg, const sensor_msgs::ImageConstPtr &depth_msg){

    cv_bridge::CvImagePtr cv_ptr1, cv_ptr2;
    try
    {
        cv_ptr1 = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e1)
    {
        ROS_ERROR("cv_bridge exception: %s", e1.what());
        return;
    }
    try
    {
        cv_ptr2 = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    }
    catch (cv_bridge::Exception &e2)
    {
        ROS_ERROR("cv_bridge exception: %s", e2.what());
        return;
    }
    cv::Mat rgb_image = cv_ptr1->image;
    cv::Mat depth_image = cv_ptr2->image;
    cv::Mat cv_mCurrImgC;
    cv::Mat cv_mCurrDepth;
    rgb_image.copyTo(cv_mCurrImgC);
    depth_image.copyTo(cv_mCurrDepth);


    // cv::imshow("rgb",rgb_image);
    cv::waitKey(1);

    //
    if(bFlag){
        Test(rgb_image, cv_mPreImgC, _v_detected_person);
    }
    //

    std_msgs::Int32 msg;
    if(bDynamic_flag)msg.data = 1;
    else msg.data = 0;
     
    _pub.publish(msg);

    cv_mPreImgC = cv_mCurrImgC;
    cv_mPreDepth = cv_mCurrDepth;
    bFlag = true;
}
void NodeServer::callback2(const yolov7_ros::ObjectsStamped& detected_result){
    int nObject_cnt = detected_result.objects.size();
    _v_detected_person.clear();
    for(int i = 0; i < nObject_cnt; i++){
        if(detected_result.objects[i].label == "person")_v_detected_person.push_back(detected_result.objects[i]);
    }
    //std::cout << v_detected_person.size() <<std::endl;
}
void Test(cv::Mat imgC, cv::Mat imgP, std::vector<yolov7_ros::Object> detected_person){
    cv::Mat imgPg, imgCg;
    cv::cvtColor(imgP, imgPg, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgC, imgCg, cv::COLOR_BGR2GRAY);
    std::vector<cv::Scalar> colors;
    // cv::RNG rng;
    // for(int i = 0; i < 100; i++)
    // {
    //     int r = rng.uniform(0, 256);
    //     int g = rng.uniform(0, 256);
    //     int b = rng.uniform(0, 256);
    //     colors.push_back(cv::Scalar(r,g,b));
    // }
    
    cv::goodFeaturesToTrack(imgPg, p0, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
    if(p0.size() == 0){
        return;
    }
    cv::Mat mask= cv::Mat::zeros(imgP.size(), imgP.type());

    std::vector<uchar> status;
    std::vector<float> err;

    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
    cv::calcOpticalFlowPyrLK(imgPg, imgCg, p0, p1, status, err, cv::Size(15,15), 2, criteria);
    std::vector<cv::Point2f> good_new;

    std::vector<Person> vPerson;
    for(int i = 0; i < detected_person.size(); i++){
        Person Temp(detected_person[i]);
        vPerson.push_back(Temp);
    }
    //p0->p1로 가는 벡터 필요
    // cv::Mat frame;
    // imgC.copyTo(frame);
    
    cv::Point2f BackgroundFlow(0,0);
    int nBGflowNum = 0;
    for(uint i = 0; i < p0.size(); i++)
    {
        // Select good points
        if(status[i] == 1) {
            bool personFlag = false;
            good_new.push_back(p1[i]);
            // Draw the tracks
            //cv::line(mask,p1[i], p0[i], colors[i], 2);
            //cv::circle(frame, p1[i], 5, colors[i], -1);

            for(int j = 0; j < vPerson.size(); j++){
                if(vPerson[j].InPerson(p1[i])){
                    personFlag = true;
                    vPerson[j].FlowSuming(p1[i]-p0[i]);
                }
            }

            if(!personFlag){
                BackgroundFlow += (p1[i]-p0[i]);
                nBGflowNum ++;
            }

        }
    }
    // std::cout << "BackGround" << "(Num: " << nBGflowNum << ") : " << BackgroundFlow << "\n";
    // for(int i = 0; i < vPerson.size(); i++){
    //     std::cout << "person" << i + 1 << "(Num: " << vPerson[i].FeatureNum() << ") : " << vPerson[i].FlowSumVector() << "\n";
    // }

    //AVG
    BackgroundFlow /= nBGflowNum;
    
    //criterion
    double dFlowMag = std::sqrt(std::pow(BackgroundFlow.x,2) + std::pow(BackgroundFlow.y,2));
    
    //dec
    for(int i = 0; i < vPerson.size(); i ++){
        vPerson[i].SetBackgroundFlow(BackgroundFlow, dFlowMag);
        vPerson[i].Calculate();
    }
    std::cout << "BackGround" << ": " << BackgroundFlow << "\n";
    for(int i = 0; i < vPerson.size(); i++){
        std::cout << "person" << i + 1 << ": " << vPerson[i].FlowAVGvector() << "\n";
    } 
    
    // Display the demo
    // cv::Mat imG;

    //cv::add(frame, mask, imG);
    cv::Mat result;
    imgC.copyTo(result);
    bDynamic_flag = false;
    for(int i = 0; i < vPerson.size(); i++){
        if(vPerson[i].IsDynamic()){
            cv::circle(result,vPerson[i].Center(), 5, cv::Scalar(0, 0, 255), -1, -1, 0);
            bDynamic_flag = true;
        }else{
            cv::circle(result,vPerson[i].Center(), 5, cv::Scalar(255, 0, 0), -1, -1, 0);
        }
    }
    cv::imshow("result",result);
    //imshow("flow", imG);
    cv::waitKey(1);
 
    // Update the previous frame and previous points
    //p0 = good_new;

}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "Dynamci_Ojbect_Detection");
    NodeServer subpub;
    ros::spin();
    return 0;
}