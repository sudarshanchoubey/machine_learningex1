package linearRegression;
import Jama.*;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.knowm.xchart.Chart_XY;
import org.knowm.xchart.Series_XY.ChartXYSeriesRenderStyle;
import org.knowm.xchart.*;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.internal.style.Styler.LegendPosition;



/** Linear Regression assignment **/

public class LinearRegression {
	public static double computeCost(Matrix X, Matrix y, Matrix theta) {
		int m = X.getRowDimension();
		Matrix temp = null;
		Matrix thetaT = null;
		Matrix XTint = null;
		Matrix H = new Matrix(m, 1);
		for (int i = 0;i < m;i++) {
			thetaT = theta.transpose();
			XTint = X.getMatrix(i, i, 0, X.getColumnDimension() - 1);
			XTint = XTint.transpose();
			temp = thetaT.times(XTint);
			H.set(i,0,temp.get(0,0));
		}
		H = H.minusEquals(y);
		H = H.arrayTimesEquals(H);
		double result = 0;
		for (int i = 0;i < m;i++) {
			result = result + H.get(i,0);
		}
		return result/(2 * m);
	}
	public static Matrix gradientDescent(Matrix X, Matrix y, Matrix theta, double alpha, int iterations) {
		int m = X.getRowDimension();
		Matrix J_History = new Matrix(iterations, 1);
		Matrix temp1 = null;
		Matrix thetaT = null;
		Matrix XTint = null;
		Matrix temp = null;
		Matrix temp2 = null;
		int thetaRowDim = 0;
		double hterm = 0;
		for(int i = 0; i < iterations; i++) {
			thetaRowDim = theta.getRowDimension();
			temp = new Matrix(thetaRowDim,1);
			for (int j = 0;j < thetaRowDim; j++) {
				hterm = 0;
				for (int k = 0;k < m;k++) {
					thetaT = theta.transpose();
					XTint = X.getMatrix(k, k, 0, X.getColumnDimension() - 1);
					XTint = XTint.transpose();
					temp1 = thetaT.times(XTint);
					temp2 = temp1.minus(y.getMatrix(k,k,0,0));
					temp2 = temp2.times(X.getMatrix(k,k,j,j));
					hterm = hterm + temp2.get(0,0);
				}
				temp.set(j,0,(alpha/m) * hterm);
			}
			theta = theta.minusEquals(temp);
			J_History.set(i,0,computeCost(X, y, theta));
		}
		J_History.print(6,3);
		return theta;
	}
	public static void main (String argv[]) throws Exception{
		InputStream is = null;
		InputStreamReader isr = null;
		BufferedReader br = null;
		Matrix m = null;
		
		is = new FileInputStream("data/ex1data1.txt");
		isr = new InputStreamReader(is);
		br = new BufferedReader(isr);
		try {
			m = Matrix.read(br);
		} catch(Exception e) {
			e.printStackTrace();
		}
		m.print(6,3);
		int nrow = m.getRowDimension();
		int ncol = m.getColumnDimension();
		Matrix X = m.getMatrix(0,nrow -1,0, 0);
		Matrix y = m.getMatrix(0,nrow - 1, 1, 1);
		double []xData = X.getColumnPackedCopy();
		double []yData = y.getColumnPackedCopy();
		Chart_XY chart = new ChartBuilder_XY().width(800).height(600).build();
		 
	    // Customize Chart
	    chart.getStyler().setDefaultSeriesRenderStyle(ChartXYSeriesRenderStyle.Scatter);
	    chart.getStyler().setChartTitleVisible(false);
	    chart.getStyler().setLegendPosition(LegendPosition.InsideSW);
	    chart.getStyler().setMarkerSize(16);
	    chart.addSeries("Scattered", xData, yData);
		new SwingWrapper(chart).displayChart();
		X.print(6,3);
		y.print(6,3);
		double[][] augX = new double[nrow][2];
		for (int i = 0; i < nrow; i++) {
			augX[i][0] = 1;
			augX[i][1] = X.get(i,0);
		}
		X = new Matrix(augX);
		X.print(6,3);
		Matrix theta = new Matrix(2, 1);
		theta.print(6,3);
		double cost = computeCost(X, y, theta);
		int iterations = 1500;
		double alpha = 0.01;
		theta = gradientDescent(X, y, theta, alpha, iterations);
		System.out.println("theta is :");
		theta.print(6,3);
   }
}
