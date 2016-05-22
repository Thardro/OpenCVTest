import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;

public class GoalRecognition {
	static String imagePath = "2016-04-08-095456.png";
	static String outputPath = "output.jpg";
	static Scalar lowerHSVBound = new Scalar(0, 62, 57, 0);
	static Scalar upperHSVBound = new Scalar(95, 255, 255, 0);
	//static final double OFFSET_ZERO = -47.5;
	static final double FOCAL_LENGTH = 0.5 * 360 / Math.tan(1.17 / 2); //TODO replace 1.17 with horizontal FOV
	static final double CENTER_X = 179.5, CENTER_Y = 239.5;
	static final double SNOUT_HEIGHT = 10.0 / 12, SNOUT_X_OFFSET = 9.25 / 12, CAMERA_X_OFFSET = 8.0 / 12, CAMERA_Y_OFFSET = 4.0 / 12; //Robot measurements in feet
	static final double GOAL_HEIGHT = 8.08; //Field measurements in feet
	
	static Mat contoursFrame;
	static Mat stencil;
	
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		//Creating u shape to match with the goal
		stencil = new Mat(8, 1, CvType.CV_32SC2);
		stencil.put(0, 0, new int[]{/*p1*/32, 0, /*p2*/ 26, 76, /*p3*/ 184, 76, /*p4*/ 180, 0, /*p5*/ 203, 0, 
				/*p6*/ 212, 100, /*p7*/ 0, 100, /*p8*/ 9, 0});
		
		long startTime = System.currentTimeMillis();
		Mat image = Imgcodecs.imread(imagePath);
		
		Mat filteredFrame = filterImageHSV(image);
		Imgcodecs.imwrite("filtered.jpg", filteredFrame);
		 
		Mat erodedFrame = erodeImage(filteredFrame);
		Imgcodecs.imwrite("eroded.jpg", erodedFrame);
		
		Mat dilatedFrame = dilateImage(erodedFrame);
		Imgcodecs.imwrite("dilated.jpg", dilatedFrame);
		
		List<MatOfPoint> contours = getContours(dilatedFrame);
	    
	    contoursFrame = dilatedFrame.clone();
	    Imgproc.cvtColor(contoursFrame, contoursFrame, Imgproc.COLOR_GRAY2BGR);
	    
	    Imgproc.drawContours(contoursFrame, contours, -1, new Scalar(255, 255, 0), 1);
	    
	    //Finding most similar contour to the desired target
	    MatOfPoint target = processContours(contours);
	    
	    //Creating a convex hull around the target and displaying it
	    Point[] targetConvexHull = calculateConvexHull(target);
	    
	    outputOverlayImage(image, contoursFrame, targetConvexHull);
	    
	    double horizontalOffset = getHorizontalOffset(targetConvexHull);
	    double verticalOffset = getVerticalOffset(targetConvexHull);
	    
	    double distance = getDistanceToGoal(horizontalOffset, verticalOffset, 0.9026);
	    
	    double robotAngle = getRobotRotation(horizontalOffset, verticalOffset, 0.9026);
	    //double angleApprox = approximateRobotRotation(horizontalOffset, verticalOffset, 0.9026);
	    System.out.println("Elapsed Time: " + (System.currentTimeMillis() - startTime) + "ms");
	    System.out.println("Robot Offset: " + robotAngle + ", Distance: " + distance);
	    
	}
	
	//Outputting an image with overlaid contours and convex hull on target
	private static void outputOverlayImage(Mat image, Mat contoursFrame, Point[] targetConvexHull) {
		//Overlaying target onto original image
		Imgproc.line(contoursFrame, targetConvexHull[0], targetConvexHull[1], new Scalar(255, 255, 255), 2);
	    Imgproc.line(contoursFrame, targetConvexHull[1], targetConvexHull[2], new Scalar(255, 255, 255), 2);
	    Imgproc.line(contoursFrame, targetConvexHull[2], targetConvexHull[3], new Scalar(255, 255, 255), 2);
	    Imgproc.line(contoursFrame, targetConvexHull[3], targetConvexHull[0], new Scalar(255, 255, 255), 2);
	    
	    Mat output = new Mat();
	    Core.bitwise_or(contoursFrame, image, output);
	    
	    Imgcodecs.imwrite(outputPath, output);
	}
	
	private static Mat filterImageHSV(Mat image) {
    	Mat hsvFrame = new Mat();
		Mat filteredFrame = new Mat();
		
		//Converting to HSV and filtering to a binary image
		Imgproc.cvtColor(image, hsvFrame, Imgproc.COLOR_BGR2HSV);
		Core.inRange(hsvFrame, lowerHSVBound, upperHSVBound, filteredFrame);
		filteredFrame.convertTo(filteredFrame, CvType.CV_8UC1);
		
		return filteredFrame;
    }
	
	private static Mat erodeImage(Mat image) {
		Mat output = new Mat();
		Imgproc.erode(image, output, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2,2)));
		
		return output;
	}
	
	private static Mat dilateImage(Mat image) {
		Mat output = new Mat();
		Imgproc.dilate(image, output, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2,2)));
		
		return output;
	}
    
    private static List<MatOfPoint> getContours(Mat image) {
    	List<MatOfPoint> contours = new ArrayList<MatOfPoint>();    
	    Imgproc.findContours(image, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
	    
	    return contours;
    }
	
	private static MatOfPoint processContours(List<MatOfPoint> contours) {
		double[] similarities = new double[contours.size()];
		for(int i = 0; i < contours.size(); i++) {
			MatOfPoint currentContour = contours.get(i);
						
			//Filtering out small contours
			if(Imgproc.contourArea(currentContour) > 400) {
				//Calculating similarity to the u shape of the goal
				double similarity = Imgproc.matchShapes(currentContour, stencil, Imgproc.CV_CONTOURS_MATCH_I3, 0);
				//System.out.println(similarity);
				if(similarity < 20) {
					similarities[i] = similarity;
				}
				else similarities[i] = 1000;
			}
			else {
				similarities[i] = 1000;
			}
		}
		
		//Finding 2 most similar of the contours, lower similarity is better
		//2 targets found as up to two goals could be in vision
		int mostSimilarGoals[] = {-1, -1};
		for(int i = 0; i < similarities.length; i++) {
			if(similarities[i] != 1000) {
				if(similarities[i] < ((mostSimilarGoals[1] == -1)? 1000: similarities[mostSimilarGoals[1]])) {
					if(similarities[i] < ((mostSimilarGoals[0] == -1)? 1000: similarities[mostSimilarGoals[0]])) {
						mostSimilarGoals[1] = mostSimilarGoals[0];
						mostSimilarGoals[0] = i;
					}
					else {
						mostSimilarGoals[1] = i;
					}
				}
			}
		}
		
		//Find widest of the goals if 2 were detected
		int mostSimilar = 0;
		if(mostSimilarGoals[1] != -1) {
			Point[][] convexHulls = {calculateConvexHull(contours.get(mostSimilarGoals[0])),
					calculateConvexHull(contours.get(mostSimilarGoals[1]))};
			double[] widths = {convexHulls[0][2].y - convexHulls[0][1].y,
					convexHulls[1][2].y - convexHulls[1][1].y};
			
			mostSimilar = (widths[0] > widths[1])? 0 : 1;
		}
			
		MatOfPoint targetContour;
		if(mostSimilarGoals[mostSimilar] == -1) {
			System.out.println("No similar contour found");
			targetContour = new MatOfPoint();
		}
		else {
			targetContour = contours.get(mostSimilarGoals[mostSimilar]);
		}
		
		return targetContour;
	}
	
	//Calculating the convex hull of a roughly rectangular contour
	private static Point[] calculateConvexHull(MatOfPoint contour) {
		Point[] targetPoints = contour.toArray();
	    Point[] convexHull = new Point[4];
	    convexHull[0] = new Point(10000, 10000);
	    convexHull[1] = new Point(0, 10000);
	    convexHull[2] = new Point(0, 0);
	    convexHull[3] = new Point(10000, 0);
	    
	    //Iterating through all points in the contour to find farthest in each direction
	    for(int i = 0; i < targetPoints.length; i++) {
	    	Point currentPoint = targetPoints[i];
	    	if (convexHull[0].x + convexHull[0].y > currentPoint.x + currentPoint.y) convexHull[0] = currentPoint;
			if (convexHull[1].y - convexHull[1].x > currentPoint.y - currentPoint.x) convexHull[1] = currentPoint;
			if (convexHull[2].x + convexHull[2].y < currentPoint.x + currentPoint.y) convexHull[2] = currentPoint;
			if (convexHull[3].x - convexHull[3].y > currentPoint.x - currentPoint.y) convexHull[3] = currentPoint;
	    }
	    
	    return convexHull;
	}
	
	private static double getHorizontalOffset(Point[] convexHull) {
		double pixelOffset = 0.5 * (convexHull[1].y + convexHull[2].y) - CENTER_X;
		
		double angleOffset = Math.atan2(pixelOffset, FOCAL_LENGTH);
		
		return angleOffset;
	}
	
	private static double getVerticalOffset(Point[] convexHull) {
		double pixelOffset = 0.5 * (convexHull[1].x + convexHull[2].x) - CENTER_Y;
		
		double angleOffset = Math.atan2(pixelOffset, FOCAL_LENGTH);
		
		return angleOffset;
	}
	
	private static double getCameraDistanceToGoal(double angleOffset, double snoutAngle) {
		double cameraHeight = Math.sin(snoutAngle) * CAMERA_X_OFFSET + Math.cos(snoutAngle) * CAMERA_Y_OFFSET + SNOUT_HEIGHT;
		//double cameraHorizontalOffset = Math.cos(snoutAngle) * CAMERA_X_OFFSET - Math.sin(snoutAngle) * CAMERA_Y_OFFSET;
		double heightToGoal = GOAL_HEIGHT - cameraHeight;
		double angleToGoal = snoutAngle + angleOffset;
		
		double distanceToGoal = heightToGoal / Math.tan(angleToGoal);
		return distanceToGoal;
	}
	
	private static double getDistanceToGoal(double horizontalAngleOffset, double verticalAngleOffset, double snoutAngle) {
		double cameraDistance = getCameraDistanceToGoal(verticalAngleOffset, snoutAngle);
		double cameraToCenter = Math.cos(snoutAngle) * CAMERA_X_OFFSET - Math.sin(snoutAngle) * CAMERA_Y_OFFSET + SNOUT_X_OFFSET;
		
		double centerToGoal = Math.sqrt(Math.pow(cameraToCenter, 2) + Math.pow(cameraDistance, 2) -
				2 * cameraToCenter * cameraDistance * Math.cos(Math.PI - horizontalAngleOffset));
		
		return centerToGoal;
	}
	
	private static double getRobotRotation(double horizontalAngleOffset, double verticalAngleOffset, double snoutAngle) {
		double cameraHorizontalOffset = Math.cos(snoutAngle) * CAMERA_X_OFFSET - Math.sin(snoutAngle) * CAMERA_Y_OFFSET + SNOUT_X_OFFSET;
		double distance = getCameraDistanceToGoal(verticalAngleOffset, snoutAngle);
		
		double centerToGoalAngle = Math.atan2(distance * Math.sin(horizontalAngleOffset), distance * Math.cos(horizontalAngleOffset) + cameraHorizontalOffset);
		
		return centerToGoalAngle;
	}
	/*
	private static double approximateRobotRotation(double horizontalAngleOffset, double verticalAngleOffset, double snoutAngle) {
		double cameraHorizontalOffset = Math.cos(snoutAngle) * CAMERA_X_OFFSET - Math.sin(snoutAngle) * CAMERA_Y_OFFSET + SNOUT_X_OFFSET;
		double distance = getCameraDistanceToGoal(verticalAngleOffset, snoutAngle);
		
		double centerToGoalAngle = horizontalAngleOffset * ( 1 - cameraHorizontalOffset / (2 * distance));
		
		return centerToGoalAngle;
	}
	*/
}
