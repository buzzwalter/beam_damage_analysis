#include "analysis.h"

/* Function: Analysis()
 *
 * I/O:
 * None
 *
 * Description:
 * Default Constructor
 *
 */
Analysis::Analysis(){

}
/* Function: ~Analysis
 *
 * I/O:
 * None
 *
 * Description:
 * Default Destructor
 *
 */
Analysis::~Analysis(){

}


/* Function: overDifferenceOpt
 *
 * I/O:
 * cvImg1u (first image input), cvImg2 (second image input),
 * maxDifference (Max Percent Difference between a pixel in first and second image before counted as change)
 * maxArea (Max percent of pixels counted as change before image is counted as damaged)
 * recordCount (records number of pixels over maxDifference threshold to caller)
 * lowIntensity (Threshold for low pixel values; pixels with intensity below this threshold get set to this threshold)
 *
 * Description:
 * First thresholds low pixel values, then finds the percent difference between the two images
 * if the percent difference on a pixel is above the threshold, it is set to 255
 * it is below it gets set to 0. Then it counts all non-zero values, and
 * checks if that is above the max area. If it is, return true overDifference, if not
 * return false overDifference.
 *
 * Uses open cv parallel functions.
 *
 */
bool Analysis::overDifferenceOpt(cv::Mat cvImg1u, cv::Mat cvImg2u, uint32_t maxDifference, uint32_t maxArea, uint32_t &recordCount, uint32_t lowIntensity, uint32_t maxDifferenceLow){
    //convert maxDifference and maxArea to double decimals i.e 20% to 0.20
    double mD = maxDifference / 100.0;
    double mA = maxArea / 100.0;

    //convert Matrices to floats in order to perform decimal computations
    cv::Mat cvImg1, cvImg2;
    cvImg1u.convertTo(cvImg1, CV_32F);
    cvImg2u.convertTo(cvImg2, CV_32F);
    uint32_t totalArea = cvImg1u.total();

    cv::Mat cvDiff, cvDiv, cvFinal, cvLargeLow1, cvLargeLow2;
    //Test matrices to avoid using camera
    //cv::Mat cvImg1(208, 200, CV_32F, 30.0);
    //cv::Mat cvImg2(208, 200, CV_32F, 50.0);



    //Set All pixels with a value below lowIntensity to have a value of lowIntensity
    cv::threshold(cvImg1, cvLargeLow1, lowIntensity, 255, cv::THRESH_TOZERO);
    cv::Mat mask1 = cvLargeLow1==0;
    cvLargeLow1.setTo(lowIntensity, mask1);

    //Set All pixels with a value below lowIntensity to have a value of lowIntensity
    cv::threshold(cvImg2, cvLargeLow2, lowIntensity, 255, cv::THRESH_TOZERO);
    cv::Mat mask2 = cvLargeLow2==0;
    cvLargeLow2.setTo(lowIntensity, mask2);

    //Find Percent Difference between two adjusted images, if above threshold, count
    cv::absdiff(cvLargeLow1, cvLargeLow2, cvDiff);
    cv::divide(cvDiff, cvLargeLow1, cvDiv);
    cv::threshold(cvDiv, cvFinal, mD, 255, cv::THRESH_BINARY);
    recordCount = cv::countNonZero(cvFinal);

    qDebug() << "Count is " << recordCount;
    //if number of pixels over percent diff is greater than percent area, return true
    if((double)recordCount/(double)totalArea > mA){
        qDebug() << "Percent over: " << (double)recordCount/(double)totalArea;
        return true;
    }
    return false;
}

/* Function: findSpot
 *
 * I/O:
 * pylonImage (image to analyze)
 * width (width of image)
 * height (height of image)
 * maxArea (max # of pixels over intensity threshold)
 * maxIntensity (Intensity threshold for each pixel)
 * recordSize (records size of spot over threshold)
 * recordAvgIntensity (records mean intensity of spot)
 *
 * Description:
 * Finds a connected area over an intensity threshold in an image.
 * if the connected area is over a number of pixels, will return true,
 * if not will return false.
 */
bool Analysis::findSpot(Pylon::CPylonImage pylonImage,uint32_t width,uint32_t height, uint32_t maxArea, uint32_t maxIntensity, uint32_t& recordSize, uint32_t& recordAvgIntensity){
    uint32_t rolling = 0;
    bool spot = false;
    if(pylonImage.IsValid()){
        //convert pylonImage to openCv matrix.
        cv::Mat cvImg = cv::Mat(pylonImage.GetHeight(), pylonImage.GetWidth(), CV_8U, static_cast<uint8_t *>(pylonImage.GetBuffer()));

        uint32_t idxX = 0;
        //Go through entire image until a bright connected area larger than maxArea is found, or until end of image
        while((spot == false) && (idxX < width)){
            uint32_t idxY = 0;
            while((spot == false) && (idxY < height)){
                if(cvImg.at<uint8_t>(cv::Point(idxX,idxY)) > maxIntensity){
                    //check if pixel is over threshold, if it is find the size of connected area
                    rolling = cvImg.at<uint8_t>(cv::Point(idxX,idxY));
                    uint32_t size = sizeSpot(cvImg, idxX, idxY, width, height, maxIntensity, rolling);
                    if(size > maxArea){
                        spot = true;
                        recordSize = size;
                    }
                }
                idxY++;
            }
            idxX++;
        }
        //record the average intensity, return true if a spot was found
        recordAvgIntensity = rolling;
        return spot;
    }
    qDebug() << "Invalid Image in analysis\n";
    return false;
}


/* Function: sizeSpot
 *
 * I/O:
 * cvImg (image to analyze)
 * width (width of image)
 * height (height of image)
 * (x, y) position of pixel over threshold
 * maxArea (max # of pixels over intensity threshold)
 * maxIntensity (Intensity threshold for each pixel)
 * rolling (rolling sum of intensities in spot)
 *
 * Description:
 * Finds a connected area over an intensity threshold in an image.
 * if the connected area is over a number of pixels, will return true,
 * if not will return false.
 */
int Analysis::sizeSpot(cv::Mat &cvImg, uint32_t x, uint32_t y, uint32_t width,uint32_t height, uint32_t maxIntensity, uint32_t& rolling){
    uint32_t count = 0;
    uint32_t right = x;
    //search for adjacent pixels above threshold to the right
    while((right < (width-1)) && cvImg.at<uint8_t>(y,right) > maxIntensity){
        uint32_t up = y;
        while((up < (height-1)) && cvImg.at<uint8_t>(up,right) > maxIntensity){
            up++;
            count++;
            rolling = rolling + cvImg.at<uint8_t>(up,right);
        }
        uint32_t down = y;
        while((down > 0) && cvImg.at<uint8_t>(down,right) > maxIntensity){
            down--;
            count++;
            rolling = rolling + cvImg.at<uint8_t>(down,right);
        }
        right++;
        count++;
        rolling = rolling + cvImg.at<uint8_t>(y,right);
    }

    if(x > 0){
        uint32_t left = x-1;
        //search for adjacent pixels above threshold to the left
        while((left > 0) && cvImg.at<uint8_t>(y,left) > maxIntensity){
            uint32_t up = y;
            while((up < (height-1)) && cvImg.at<uint8_t>(up,left) > maxIntensity){
                up++;
                count++;
                rolling = rolling + cvImg.at<uint8_t>(up,left);
            }
            uint32_t down = y;
            while((down > 0) && cvImg.at<uint8_t>(down,left) > maxIntensity){
                down--;
                count++;
                rolling = rolling + cvImg.at<uint8_t>(down,left);
            }
            left--;
            count++;
            rolling = rolling + cvImg.at<uint8_t>(y,left);
        }
    }
    if(count == 0){
    rolling = 0;
    }
    else{
    rolling = rolling / count;
    }
    return count;
}


/* Function: overDifference
 *
 * I/O:
 * prev (image found one frame previously)
 * curr (most recent image found)
 * width (width of the images)
 * height (height of the images)
 * maxDifference (max percent difference between pixels in the images)
 * maxArea (max number of pixels over maxDifference)
 * recordSize (records number of pixels over maxDifference to caller)
 * lowIntensity (low threshold for pixels)
 * maxDifferenceLow (max number of intensity difference for pixels below lowIntensity)
 * lowOverCount (number of pixels over threshold below lowIntensity)
 * highOverCount (number of pixels over threshold above lowIntensity)
 *
 * Description
 * Given two images, finds if the number of pixels in the same location that are differing by a certain threshold.
 * If the pixels are above lowIntensity, it finds a percent difference, and checks against maxDifference.
 * If the pixels are below lowIntensity, it finds a number difference, and checks against maxDifferenceLow.
 * If the sum of the number of pixels above their respective thresholds is greater than maxArea, return true.
 * else return false.
 */
bool Analysis::overDifference(Pylon::CPylonImage prev, Pylon::CPylonImage curr ,uint32_t width,uint32_t height, uint32_t maxDifference, uint32_t maxArea, uint32_t& recordSize, uint32_t lowIntensity, uint32_t maxDifferenceLow, uint32_t& lowOvercount, uint32_t& highOverCount){
    bool over = false;
    uint32_t numDiff = 0;
    highOverCount = 0;
    lowOvercount = 0;
    if(prev.IsValid()){
        cv::Mat cvImg = cv::Mat(prev.GetHeight(), prev.GetWidth(), CV_8U, static_cast<uint8_t *>(prev.GetBuffer()));
        if(curr.IsValid()){
            cv::Mat cvImg2 = cv::Mat(curr.GetHeight(), curr.GetWidth(), CV_8U, static_cast<uint8_t *>(curr.GetBuffer()));
            numDiff = countDifference(cvImg, cvImg2, width, height, maxDifference, lowIntensity, maxDifferenceLow, lowOvercount, highOverCount);
        }
        if((numDiff) > maxArea){
            qDebug() << "Above Threshold, " << numDiff << "Pixels changed.\n";
            over = true;
            recordSize = numDiff;
        }
    }
    else
    {
        qDebug() << "Image not valid\n";
    }
    return over;
}

/* Function: countDifference
 *
 * I/O:
 * prevFrame (image found one frame previously)
 * currFrame (most recent image found)
 * width (width of the images)
 * height (height of the images)
 * maxDifference (max percent difference between pixels in the images)
 * maxArea (max number of pixels over maxDifference)
 * recordSize (records number of pixels over maxDifference to caller)
 * lowIntensity (low threshold for pixels)
 * maxDifferenceLow (max number of intensity difference for pixels below lowIntensity)
 * lowOverCount (number of pixels over threshold below lowIntensity)
 * highOverCount (number of pixels over threshold above lowIntensity)
 *
 * Description
 * Given two images, finds if the number of pixels in the same location that are differing by a certain threshold.
 * If the pixels are above lowIntensity, it finds a percent difference, and checks against maxDifference.
 * If the pixels are below lowIntensity, it finds a number difference, and checks against maxDifferenceLow.
 * If the sum of the number of pixels above their respective thresholds is greater than maxArea, return true.
 * else return false.
 */
int Analysis::countDifference(cv::Mat prevFrame, cv::Mat currFrame,uint32_t width,uint32_t height, uint32_t maxDifference, uint32_t lowIntensity, uint32_t maxDifferenceLow, uint32_t& lowOvercount, uint32_t& highOverCount){

        uint32_t numDiff = 0;
        lowOvercount = 0;
        highOverCount = 0;
        uint32_t idxX = 0;
        uint32_t idxY;
        uint32_t pVal;
        uint32_t cVal;
        double percentdiff = maxDifference/100.0;
        while(idxX < width){
            idxY = 0;
            while(idxY < height){
                pVal = prevFrame.at<uint8_t>(cv::Point(idxX,idxY));
                cVal = currFrame.at<uint8_t>(cv::Point(idxX,idxY));
                //if either image's pixel is above low intensity, find if percent difference > threshold
                if((cVal > lowIntensity) || (pVal > lowIntensity)){
                    if(cVal > (pVal*(1 + percentdiff))){
                        highOverCount++;
                        numDiff++;
                    }
                    else if(cVal < (pVal*(1 - percentdiff))){
                        highOverCount++;
                        numDiff++;
                    }
                }
                //if both image's pixel is below low intensity, find if number difference > threshold
                else{
                    //case for small values
                    if(cVal > (pVal+maxDifferenceLow)){
                        lowOvercount++;
                        numDiff++;
                    }
                    else if((pVal >= maxDifferenceLow) && (cVal < (pVal-maxDifferenceLow))){
                        lowOvercount++;
                        numDiff++;
                    }
                }
                idxY++;
            }
            idxX++;
        }
        return numDiff;
}


/*Function: spotDiameter
 *
 *
 *
 *
 *Descrioption:
 *find and return the width and height of the beam spot 
 */
int Analysis::spotDiameter(cv::Mat frame){
    int width=0;
    int height=0;
    //use 1/e^2 of max to define width 1/e^2~=0.135 = 13.5%
    double e2 = 0.135;

    //max of each y collumn to get row of maxima, use with e2thresh to get width
    cv:Mat xMax;
    reduce(frame,xMaxArr,0,CV_REDUCE_MAX,-1);
    
    //get image max and calculate 1/e^2 threshold
    int imgMax=max(xMaxArr);
    double e2thresh = max*e2;
    
    //check each max value along the x axis, if val is > max/e^2 increment width
    for(int i = 0; i < xMax.size(); ++i) {
        if(xMax[i] >= e2Thresh){
            width++;
        }
    }
    
    //max of each x row to get collumn of maxima, use with e2thresh to get height
    cv:Mat yMax;
    reduce(frame,yMax,1,CV_REDUCE_MAX,-1);
    //check each max value along the y axis, if val is > max/e^2 increment height
    for(int i = 0; i < yMax.size(); ++i) {
        if(yMax[i] >= Thresh){
            height++;
        }
    }

    return [width, height]
}

/*Function: size Difference
 *
 *I/O 
 *
 *currFrame (most recent image found)
 *refFrameData (width and height of spot in referencence image)
 *reduceThreshold ()
 *
 *
 *
 *
 *Description
 *Compare the spot size of the current image to a previously defined reference image to determine if the beam has reduced in size since the begining
 *If the diameter of the beam spot changes/decreases by more than a given threshold %
 */
bool Analysis::sizeDifference(cv::Mat currFrame,cv::Mat refFrameData, double reduceThreshold){
    [width_c,height_c]=spotDiameter(currFrame);
    [width_r,height_r]=refFrameData;
    //get reduction threshold where function will return true
    double threshWidth=width_r*(reduceThreshold/100);
    double threshHeight=height_r*(reduceThreshold/100);
    //calculate difference
    int width_diff=width_r-width_c;
    int height_diff=height_r-height_c;


    if(width_diff>=threshWidth||height_diff>=threshHeight){
        return true;
    } else {
        return false;
    }
}



