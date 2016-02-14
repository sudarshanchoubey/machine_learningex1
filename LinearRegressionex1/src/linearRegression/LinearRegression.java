package linearRegression;
import Jama.*;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.knowm.xchart.Chart_XY;
import org.knowm.xchart.Series_XY.ChartXYSeriesRenderStyle;
import org.knowm.xchart.Series_XY;
import org.knowm.xchart.*;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.internal.style.Styler.LegendPosition;
import org.knowm.xchart.internal.style.markers.None;



/** Linear Regression assignment **/

public class LinearRegression {
	/* Computes the cost for all inputs using current theta.
	 * Cost function is summation of (theta' x X[i,:] - y[i])^2 / (2 * number of inputs)
	 */
	public static double computeCost(Matrix X, Matrix y, Matrix theta) {
		int m = X.getRowDimension();
		Matrix temp = null;
		Matrix XTint = null;
		Matrix H = new Matrix(m, 1);
		for (int i = 0;i < m;i++) {
			XTint = X.getMatrix(i, i, 0, X.getColumnDimension() - 1).transpose();
			temp = theta.transpose().times(XTint);
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
	/*
	 * Plots all observations on a chart.
	 */
	public static Chart_XY showChart(double xData[], double yData[]) {
		Chart_XY chart = new ChartBuilder_XY().width(840).height(620).build();
		 
	    chart.getStyler().setDefaultSeriesRenderStyle(ChartXYSeriesRenderStyle.Scatter);
	    chart.getStyler().setChartTitleVisible(true);
	    chart.setTitle("Profits by population");
	    chart.setXAxisTitle("population in 10,000s");
	    chart.setYAxisTitle("Profits in $10,000s");
	    chart.getStyler().setLegendPosition(LegendPosition.InsideNW);
	    chart.getStyler().setMarkerSize(16);
	    chart.addSeries("Inputs", xData, yData);
		new SwingWrapper(chart).displayChart();
		return chart;
	}
	/*
	 * Plots the observations and the line for our predictions.
	 */
	public static Chart_XY showPredictionChart(Chart_XY chart,double xData[], double yData[]) {
		Series_XY series= chart.addSeries("Predictions", xData, yData);
		series.setChartXYSeriesRenderStyle(ChartXYSeriesRenderStyle.Line);
		series.setMarker(new None());
		new SwingWrapper(chart).displayChart();
		return chart;
	}
	/*
	 * Run gradient descent for required number of iterations.
	 */
	public static Matrix gradientDescent(Matrix X, Matrix y, Matrix theta, double alpha, int iterations) {
		int m = X.getRowDimension();
		Matrix J_History = new Matrix(iterations, 1);
		Matrix temp1 = null;
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
					XTint = X.getMatrix(k, k, 0, X.getColumnDimension() - 1);
					temp1 = theta.transpose().times(XTint.transpose());
					temp2 = temp1.minus(y.getMatrix(k,k,0,0));
					temp2 = temp2.times(X.getMatrix(k,k,j,j));
					hterm = hterm + temp2.get(0,0);
				}
				temp.set(j,0,(alpha/m) * hterm);
			}
			theta = theta.minusEquals(temp);
			J_History.set(i,0,computeCost(X, y, theta));
		}
		return theta;
	}
	public static Matrix readInputMatrix() throws Exception {
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
		return m;
	}
	public static void main (String argv[]) throws Exception{
		Matrix matrix = readInputMatrix();
		int nrow = matrix.getRowDimension();
		Matrix X = matrix.getMatrix(0,nrow -1,0, 0);
		Matrix y = matrix.getMatrix(0,nrow - 1, 1, 1);
		double []xData = X.getColumnPackedCopy();
		double []yData = y.getColumnPackedCopy();
		Chart_XY chart = showChart(xData, yData);
		double[][] augX = new double[nrow][2];
		for (int i = 0; i < nrow; i++) {
			augX[i][0] = 1;
			augX[i][1] = X.get(i,0);
		}
		X = new Matrix(augX);
		Matrix theta = new Matrix(2, 1);
		double cost = computeCost(X, y, theta);
		System.out.println("Starting Cost is " + Double.toString(cost));
		int iterations = 1500;
		double alpha = 0.01;
		theta = gradientDescent(X, y, theta, alpha, iterations);
		double[] prediction = null;
		double[] inp_for_chart = null;
		Matrix P1 = X.times(theta);
		prediction = P1.getColumnPackedCopy();
		P1 = X.getMatrix(0, nrow -1, 1, 1);
		inp_for_chart = P1.getColumnPackedCopy();
		chart = showPredictionChart(chart, inp_for_chart, prediction);
		cost = computeCost(X, y, theta);
		System.out.println("Final Cost is " + Double.toString(cost));
		System.out.println("theta is :");
		theta.print(6,3);
   }
}
