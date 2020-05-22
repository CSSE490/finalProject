import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;

public class Training {
	
	public static double[][] imageData;
	public static double[][] labelData;
	public static int numberOfImages;
	public static int numberOfLabels;
	private static double[][] imageDatatest;
	private static double[][] labelDatatest;
	private static int numberOfImagesTest;
	private static int numberOfLabelsTest;

	public static void main(String[] args){

		
		System.out.println("started reading data");
		idxReader();
		System.out.println("finished reading data");

		FeedForwardNetwork n = new FeedForwardNetwork(784, 125, 1, 10);
//		double[][] inputs = {		
//				{0,0},
//				{0,1},
//				{1,0},
//				{1,1}
//		}; 
//		double[][] desiredOutput = {{0},{1},{1},{0}};		
		n.initNetwork(imageData, labelData, 0.7, 0.5);
		System.out.println("Start trainning");
		long startTime = System.currentTimeMillis();
		n.trainNetwork(20, false);
		long endTime   = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		//n.printWeights();
		System.out.println("start reading test data");
		idxReaderTest();
		System.out.println("finished reading test data");
		System.out.println("Test result:");
		startTime = System.currentTimeMillis();
		n.testNetwork(imageDatatest, labelDatatest);
		endTime   = System.currentTimeMillis();
		totalTime = endTime - startTime;
		System.out.printf("test time used: %f s\n", totalTime / 1000.0);

	}
	
	
	public static void idxReader()
	/*the data file should be placed into the same folder as the training.jave file.*/ 
	{
		FileInputStream inImage = null;
        FileInputStream inLabel = null;

        //String inputImagePath = "/Users/apple/Desktop/DIGITREC/train-images-idx3-ubyte";
        String inputImagePath = Training.class.getResource("train-images-idx3-ubyte").getPath();
        //System.out.println(Training.class.getResource("train-images-idx3-ubyte").getPath());
        //String inputLabelPath = "/Users/apple/Desktop/DIGITREC/train-labels-idx1-ubyte";
        String inputLabelPath = Training.class.getResource("train-labels-idx1-ubyte").getPath();

        String outputPath = "/Users/apple/Desktop/DIGITREC/";

        //int[] hashMap = new int[10]; 

        try {
            inImage = new FileInputStream(inputImagePath);
            inLabel = new FileInputStream(inputLabelPath);

            int magicNumberImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            numberOfImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfRows  = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfColumns = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());

            int magicNumberLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());
            numberOfLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());

            //BufferedImage image = new BufferedImage(numberOfColumns, numberOfRows, BufferedImage.TYPE_INT_ARGB);
            int numberOfPixels = numberOfRows * numberOfColumns;
            int[] imgPixels = new int[numberOfPixels];
            
            
            //numberOfImages = 30;
            
            imageData = new double[numberOfImages][numberOfPixels];
            labelData = new double[numberOfLabels][10];

            for(int i = 0; i < numberOfImages; i++) {

                //if(i % 10000 == 0) {System.out.println("Number of images extracted: " + i);}
            	double[] imageInts = new double[numberOfPixels];
                for(int p = 0; p < numberOfPixels; p++) {
                    int value = inImage.read() > 50 ? 1 : 0;
//                    System.out.println(value);
                	imageInts[p] = value;
                }
//                if(i < 10) {
//            		for(int j = 0; j < numberOfColumns; j++) {
//            			for(int k = 0; k < numberOfRows; k++) {
//            				System.out.print(imageInts[j * numberOfRows + k]);
//            			}
//            			System.out.println();
//            		}
//            		System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
//            	}
                
                imageData[i] = imageInts;
                int label = inLabel.read();
                labelData[i][label] = 1; 

//                image.setRGB(0, 0, numberOfColumns, numberOfRows, imgPixels, 0, numberOfColumns);
//                
//                ByteArrayOutputStream baos = new ByteArrayOutputStream();
//                ImageIO.write( image, "png", baos );
//                baos.flush();
//                byte[] imageInByte = baos.toByteArray();
//                imageData.add(imageInByte);
//                baos.close();
//
//                int label = inLabel.read();
//                labelData.add(label);
//
//                hashMap[label]++;
                //File outputfile = new File(outputPath + label + "_0" + hashMap[label] + ".png");

                //ImageIO.write(image, "png", outputfile);
            }
//            System.out.println(numberOfImages + " " + numberOfLabels);

        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } finally {
            if (inImage != null) {
                try {
                    inImage.close();
                } catch (IOException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
            if (inLabel != null) {
                try {
                    inLabel.close();
                } catch (IOException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
        }
	}
        
        
    public static void idxReaderTest()
	{
		FileInputStream inImage = null;
        FileInputStream inLabel = null;

        //String inputImagePath = "/Users/apple/Desktop/DIGITREC/t10k-images-idx3-ubyte";
        String inputImagePath = Training.class.getResource("t10k-images-idx3-ubyte").getPath();

//        String inputLabelPath = "/Users/apple/Desktop/DIGITREC/t10k-labels-idx1-ubyte";
        String inputLabelPath = Training.class.getResource("t10k-labels-idx1-ubyte").getPath();
        

        String outputPath = "/Users/apple/Desktop/DIGITREC/";

        //int[] hashMap = new int[10]; 

        try {
            inImage = new FileInputStream(inputImagePath);
            inLabel = new FileInputStream(inputLabelPath);

            int magicNumberImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            numberOfImagesTest = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfRows  = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfColumns = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());

            int magicNumberLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());
            numberOfLabelsTest = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());

            //BufferedImage image = new BufferedImage(numberOfColumns, numberOfRows, BufferedImage.TYPE_INT_ARGB);
            int numberOfPixels = numberOfRows * numberOfColumns;
            int[] imgPixels = new int[numberOfPixels];
            
            
            //numberOfImages = 30;
            
            imageDatatest = new double[numberOfImagesTest][numberOfPixels];
            labelDatatest = new double[numberOfLabelsTest][10];

            for(int i = 0; i < numberOfImagesTest; i++) {

                //if(i % 100 == 0) {System.out.println("Number of images extracted: " + i);}
            	double[] imageInts = new double[numberOfPixels];

            	for(int p = 0; p < numberOfPixels; p++) {
                    int value = inImage.read() > 50 ? 1 : 0;
//                    System.out.println(value);
                	imageInts[p] = value;
                }
//                if(i < 10) {
//            		for(int j = 0; j < numberOfColumns; j++) {
//            			for(int k = 0; k < numberOfRows; k++) {
//            				System.out.print(imageInts[j * numberOfRows + k]);
//            			}
//            			System.out.println();
//            		}
//            		System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
//            	}
                
                imageDatatest[i] = imageInts;
                int label = inLabel.read();
                labelDatatest[i][label] = 1; 
            }
//          System.out.println(numberOfImagesTest + " " + numberOfLabelsTest);


        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } finally {
            if (inImage != null) {
                try {
                    inImage.close();
                } catch (IOException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
            if (inLabel != null) {
                try {
                    inLabel.close();
                } catch (IOException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
        }
	}
	
}
