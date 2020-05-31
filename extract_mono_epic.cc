/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<iomanip>
#include<ctime>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>
#include<Converter.h>

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 6)
    {
        cerr << endl << "Usage: ./mono_epic path_to_vocabulary path_to_settings path_to_sequence path_to_save (use_viewer)" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3]);
    bool use_viewer = (string(argv[5]) == "1");
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,use_viewer);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    vector<cv::Mat> all_T;
    // Main loop
    cv::Mat im;
    int state;
    int last_state = 1;
    // Record time
    clock_t begin_time = clock();
    clock_t end_time = clock();
    clock_t total_begin_time = clock();

    for(int ni=0; ni<nImages; ni++)
    {   
        // Read image from file
        im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif
        // Pass the image to the SLAM system
        cv::Mat Tcw = SLAM.TrackMonocular(im,tframe);
        state = SLAM.GetTrackingState();
        // enum eTrackingState{
        //     SYSTEM_NOT_READY=-1,
        //     NO_IMAGES_YET=0,
        //     NOT_INITIALIZED=1,
        //     OK=2,
        //     LOST=3
        // };

        // Output tracking state information
        if (state != last_state)
        {
            if (state == 2)
                {
                    if (last_state == 1)
                        cout << "Image: " << setw(6) << ni << " Initialized! " << endl;
                    else if (last_state == 3)
                        cout << "Image: " << setw(6) << ni << " Continue tracking! " << endl;
                }
            else if(state == 3)
                cout << "Image: "  << setw(6) << ni << " Tracking lost! " << endl;
            else
                cout << "Image: "  << setw(6) << ni << " Tracking state: " << state << " Last tracking state: " << last_state << endl;
        }
        last_state = state;

        if ((ni + 1) % 100 == 0)
        {
            end_time = clock();
            cout << "Image: " << setw(6) << ni + 1 << " Tracking state: " << state << " Tracking time: " << double(end_time - begin_time) / CLOCKS_PER_SEC << endl;
            begin_time = clock();

        }
        // Save camera position
        all_T.push_back(Tcw);


#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        // if(ttrack<T)
        //     usleep((T-ttrack)*1e6);
        if(ttrack<0.3)
            usleep((0.3 - ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    clock_t total_end_time = clock();
    cout << "Total used time: " << double(total_end_time - total_begin_time) / CLOCKS_PER_SEC << endl;
    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    string prefix = string(argv[4]);
    SLAM.SaveKeyFrameTrajectoryTUM(prefix + "_keyFrame.txt");
    
    string filename = prefix + "_Frame.txt";
    cout << endl << "Saving frame trajectory to " << filename << " ..." << endl;
    ofstream f;
    f.open(filename.c_str());
    f << fixed;
    for(int ni = 0; ni < nImages; ni++)
    {
        cv::Mat Tcw = all_T[ni];
        if(Tcw.empty())
        {
            Tcw = cv::Mat::eye(4,4,CV_32F);
        }
        cv::Mat R = Tcw.rowRange(0,3).colRange(0,3).t();
        vector<float> q = ORB_SLAM2::Converter::toQuaternion(R);
        cv::Mat t = -R * Tcw.rowRange(0,3).col(3);
        f << setprecision(6) << vTimestamps[ni] << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;

    string map_filename = prefix + "_mapPoint.txt";
    cout << endl << "Saving map points to " << map_filename << " ..." << endl;
    ofstream f2;
    f2.open(map_filename.c_str());
    f2 << fixed;
    ORB_SLAM2::Map* mpMap = SLAM.get_map();
    const vector<ORB_SLAM2::MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        if(vpMPs[i]->isBad())
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        f2 << setprecision(7) << " " << pos.at<float>(0) << " " << pos.at<float>(1) << " " << pos.at<float>(2) << endl;
    }
    f2.close();
    cout << endl << "map points saved!" << endl;

    string refmap_filename = prefix + "_refMapPoint.txt";
    cout << endl << "Saving reference map points to " << refmap_filename << " ..." << endl;
    ofstream rf;
    rf.open(refmap_filename.c_str());
    rf << fixed;
    vector<ORB_SLAM2::KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),ORB_SLAM2::KeyFrame::lId);
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        ORB_SLAM2::KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;

        rf << "KeyFrame: " << endl;
        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = ORB_SLAM2::Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        rf << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

        set<ORB_SLAM2::MapPoint*> spRefMPs = pKF->GetMapPoints();
        for(set<ORB_SLAM2::MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
        {
            if((*sit)->isBad())
                continue;
            cv::Mat pos = (*sit)->GetWorldPos();
            rf << setprecision(7) << " " << pos.at<float>(0) << " " << pos.at<float>(1) << " " << pos.at<float>(2) << endl;
        }
    }
    rf.close();
    cout << endl << "reference map points saved!" << endl;
    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    // string s0;
    // getline(f,s0);
    // getline(f,s0);
    // getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}
