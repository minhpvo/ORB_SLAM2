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
    if(argc != 5)
    {
        cerr << endl << "Usage: ./mono_epic path_to_vocabulary path_to_settings path_to_sequence path_to_save" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3]);
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

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
    for(int ni=0; ni<nImages; ni++)
    {   
        // Record time
        clock_t begin_time = clock();

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

        clock_t end_time = clock();
        if ((ni + 1) % 100 == 0)
            cout << "Image: " << setw(6) << ni + 1 << " Tracking state: " << state << " Tracking time: " << double(end_time - begin_time) / CLOCKS_PER_SEC << endl;

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
	cout << "t: " << ttrack << endl;
        //if(ttrack<T/60)
        //    usleep((T/60-ttrack)*1e6);
        if(ttrack<T)
            usleep(0.05*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

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
    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

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
