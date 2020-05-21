import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;

public class PrintTest {
	
	public static double[][] imageData;
	public static double[][] labelData;
	public static int numberOfImages;
	public static int numberOfLabels;
	private static double[][] imageDatatest;
	private static double[][] labelDatatest;
	private static int numberOfImagesTest;
	private static int numberOfLabelsTest;

	public static void main(String[] args){

		
		idxReader();
		//FeedForwardNetwork n = new FeedForwardNetwork(784, 125, 1, 10);
//		double[][] inputs = {		
//				{0,0},
//				{0,1},
//				{1,0},
//				{1,1}
//		}; 
//		double[][] desiredOutput = {{0},{1},{1},{0}};		
//		n.initNetwork(imageData, labelData, 0.7, 0.5);
//		n.trainNetwork(20, false);
//		n.printWeights();
		
		
		
		//idxReaderTest();
//		n.testNetwork(imageDatatest, labelDatatest);
	}
	
	
	public static void idxReader()
	{
		FileInputStream inImage = null;
        FileInputStream inLabel = null;

        String inputImagePath = "/Users/apple/Desktop/DIGITREC/train-images-idx3-ubyte";
        String inputLabelPath = "/Users/apple/Desktop/DIGITREC/train-labels-idx1-ubyte";

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
            int count7 = 0;
            for(int i = 0; i < numberOfImages; i++) {

                //if(i % 10000 == 0) {System.out.println("Number of images extracted: " + i);}
            	double[] imageInts = new double[numberOfPixels];
                for(int p = 0; p < numberOfPixels; p++) {
                    int value = inImage.read() > 50 ? 1 : 0;
//                    System.out.println(value);
                	imageInts[p] = value;
                }
                
                
                imageData[i] = imageInts;
                int label = inLabel.read();
                
                if(i > 59000) {
            		for(int j = 0; j < numberOfColumns; j++) {
            			for(int k = 0; k < numberOfRows; k++) {
            				System.out.print(imageInts[j * numberOfRows + k]);
            			}
            			System.out.println();
            		}
                    System.out.println(label + "^^^");
            		System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            		
            	}
                

                labelData[i][label] = 1;
                if(label == 7) {
                	count7++;
                }

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
            System.out.println(count7 + " number of 7" );
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

        String inputImagePath = "/Users/apple/Desktop/DIGITREC/t10k-images-idx3-ubyte";
        String inputLabelPath = "/Users/apple/Desktop/DIGITREC/t10k-labels-idx1-ubyte";

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
            int count7 = 0;
            for(int i = 0; i < numberOfImagesTest; i++) {

                //if(i % 100 == 0) {System.out.println("Number of images extracted: " + i);}
            	double[] imageInts = new double[numberOfPixels];

            	for(int p = 0; p < numberOfPixels; p++) {
                    int value = inImage.read() > 50 ? 1 : 0;
//                    System.out.println(value);
                	imageInts[p] = value;
                }
//                if(i > 9000) {
//            		for(int j = 0; j < numberOfColumns; j++) {
//            			for(int k = 0; k < numberOfRows; k++) {
//            				System.out.print(imageInts[j * numberOfRows + k]);
//            			}
//            			System.out.println();
//            		}
//            		System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    " + i);
//            	}
                
                imageDatatest[i] = imageInts;
                int label = inLabel.read();
                labelDatatest[i][label] = 1; 
                if(label == 7) {
                	count7++;
                }
                System.out.println("label: " + label);
            }
            System.out.println("number of count7: " + count7);
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
