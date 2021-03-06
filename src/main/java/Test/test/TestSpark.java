package Test.test;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import java.util.regex.Pattern;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;

import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.*;
import org.apache.hadoop.hdfs.server.namenode.FSEditLogLoader.PositionTrackingInputStream;
import org.apache.spark.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.random.RandomRDDs;
import org.apache.spark.rdd.PairRDDFunctions;
import org.apache.spark.mllib.fpm.AssociationRules;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowthModel;
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset;
import org.apache.spark.rdd.*;

public class TestSpark {
	
	// CLasse permettant de parser des réels contenus dans un fichier texte
	private static class ParsePoint implements Function<String, Vector> {
	    private static final Pattern SPACE = Pattern.compile("\t");

	    //@Override
	    public Vector call(String line) {
	      String[] tok = SPACE.split(line);
	      double[] point = new double[tok.length];
	      for (int i = 0; i < tok.length; ++i) {
	        point[i] = Double.parseDouble(tok[i]);
	      }
	      return Vectors.dense(point);
	    }
	  }
	
	public static void test0() {
		String logFile = "/Users/Alain/Desktop/Texte.txt"; // Doit être un fichier existant ; on peut aussi spécifier un fichier HDFS !

		SparkConf conf = new SparkConf().setAppName("Simple Application").setMaster("local[2]"); // sur 2 cores en local
		JavaSparkContext sc = new JavaSparkContext(conf); // Création du contexte spark

		JavaRDD<String> logData = sc.textFile(logFile).cache(); // on pesiste la RDD en mémoire vive

		long numAs = logData.filter(new Function<String, Boolean>() {
			public Boolean call(String s) { 
				return s.contains("a");
			}
		}).count(); // on compte les "a"

		long numBs = logData.filter(new Function<String, Boolean>() {
			public Boolean call(String s) { 
				return s.contains("b"); 
			}
		}).count(); // on compte les "b"

		// on affiche les résultats
		System.out.println("Lines with a: " + numAs + ", lines with b: " + numBs);
	
		sc.close();// on ferme le contexte
	}

	public static void testRDD()  {
		SparkConf conf = new SparkConf().setAppName("Test random").setMaster("local[2]");
	    JavaSparkContext jsc = new JavaSparkContext(conf);

	    //création d'un RDD de valeurs aléatoires distribuées selon une loi normale
	    JavaDoubleRDD u = RandomRDDs.normalJavaRDD(jsc, 10L, 10);
	    // On applique une fonction de transformation sur les valeurs générées
	    JavaRDD<Double> v = (JavaRDD<Double>) u.map(
		   new Function<Double, Double>() {
		     public Double call(Double x) {
		       return 1.0 + 2.0 * x;
		     }
		   });

	    // pattern d'affichage
	    Consumer style = (d) -> System.out.println(d);
	    
	    // application du pattern
	    v.collect().forEach(style);
	    
	    jsc.close();
	}
	
	public static void kMeans() {
		String logFile = "/Users/Alain/Desktop/Data/net1_500_500.txt";
	    int k = 5; // nombre de k-moyennes
	    int iterations = 100; // nombre d'itérations maximums de l'algorithme
	    int runs = 1; // nombre d'execution de l'algorithme en parallèle
	    List<String> nomAttributs =  new ArrayList<String>();

	    SparkConf conf = new SparkConf().setAppName("Simple Application").setMaster("local[2]"); // sur 2 cores en local
		JavaSparkContext jsc = new JavaSparkContext(conf);
		jsc.setLogLevel("WARN");
		JavaRDD<Vector> points = RandomRDDs.normalJavaVectorRDD(jsc, 100, 500, 4);
	    //JavaRDD<Vector> points = lectureFichier("/Users/Alain/Desktop/Data/net1_100_500.txt", jsc, nomAttributs);


	    KMeansModel model = KMeans.train(points.rdd(), k, iterations, runs, KMeans.K_MEANS_PARALLEL());

	    //affichage
	    System.out.println("Cluster centers:");
	    for (Vector center : model.clusterCenters()) {
	      System.out.println(" " + center);
	    }
	    double cost = model.computeCost(points.rdd());
	    System.out.println("Cost: " + cost);

	    jsc.stop();
	    jsc.close();
	}

	public static void testKMeansRandom() {
		int k = 5;
	    int iterations = 100;
	    int runs = 1;
		
		SparkConf conf = new SparkConf().setAppName("Test random").setMaster("local[2]");
	    JavaSparkContext jsc = new JavaSparkContext(conf);

	    JavaRDD<Vector> u = RandomRDDs.normalJavaVectorRDD(jsc, 100L,10);
	    
	    KMeansModel model = KMeans.train(u.rdd(), k, iterations, runs, KMeans.K_MEANS_PARALLEL());

	    System.out.println("Cluster centers:");
	    for (Vector center : model.clusterCenters()) {
	      System.out.println(" " + center);
	    }
	    double cost = model.computeCost(u.rdd());
	    System.out.println("Cost: " + cost);
	    
	    jsc.close();
	}
	
	public static void testFPGRowth() {
		String inputFile = "/Users/Alain/Desktop/FPGrowthExample.txt";
	    double minSupport = 0.3;
	    int numPartition = -1;

	    SparkConf sparkConf = new SparkConf().setAppName("JavaFPGrowthExample").setMaster("local[2]");
	    JavaSparkContext sc = new JavaSparkContext(sparkConf);

	    JavaRDD<ArrayList<String>> transactions = sc.textFile(inputFile).map(
	      new Function<String, ArrayList<String>>() {
	        @Override
	        public ArrayList<String> call(String s) {
	          return Lists.newArrayList(s.split(" "));
	        }
	      }
	    );

	    FPGrowthModel<String> model = new FPGrowth()
	      .setMinSupport(minSupport)
	      .setNumPartitions(numPartition)
	      .run(transactions);

	    AssociationRules arules = new AssociationRules().setMinConfidence(0.8);
	    JavaRDD<AssociationRules.Rule<String>> results = arules.run((JavaRDD<FreqItemset<String>>) model.freqItemsets().toJavaRDD());
	    
	    for (FPGrowth.FreqItemset<String> s: model.freqItemsets().toJavaRDD().collect()) {
	      System.out.println("[" + Joiner.on(",").join(s.javaItems()) + "], " + s.freq());
	    }

	    for (AssociationRules.Rule<String> rule : results.collect()) {
		      System.out.println(
		        rule.javaAntecedent() + " => " + rule.javaConsequent() + ", " + rule.confidence());
		    }
	    
	    sc.stop();
	    sc.close();
	}
	
	public static void testAssociationRules() {
		SparkConf sparkConf = new SparkConf().setAppName("JavaAssociationRulesExample").setMaster("local[2]");
	    JavaSparkContext sc = new JavaSparkContext(sparkConf);

	    // $example on$
	    JavaRDD<FPGrowth.FreqItemset<String>> freqItemsets = sc.parallelize(Arrays.asList(
	      new FreqItemset<String>(new String[] {"a"}, 15L),
	      new FreqItemset<String>(new String[] {"b"}, 35L),
	      new FreqItemset<String>(new String[] {"a", "b"}, 12L)
	    ));

	    AssociationRules arules = new AssociationRules().setMinConfidence(0.8);
	    JavaRDD<AssociationRules.Rule<String>> results = arules.run(freqItemsets);

	    for (AssociationRules.Rule<String> rule : results.collect()) {
	      System.out.println(
	        rule.javaAntecedent() + " => " + rule.javaConsequent() + ", " + rule.confidence());
	    }
	}
	
	public static void associationRules() {
		SparkConf conf = new SparkConf().setAppName("Test random").setMaster("local[2]").set("spark.driver.memory", "10g");
	    JavaSparkContext jsc = new JavaSparkContext(conf);
	    jsc.setLogLevel("WARN");
	    int n = 600; // n attributes
	    long m = 100L; // m item, can be a VERY LARGE number
	    double minSupport = 0.3;
	    double minConfidence = 0.8;
	    int numPartition = -1;
	    List<String> nomAttributs =  new ArrayList<String>();

	    double debut = System.currentTimeMillis();
	    //Creation of a RDD with a centered reduced normal random distribution
	    //JavaRDD<Vector> u = RandomRDDs.normalJavaVectorRDD(jsc, m, n, 4);
	    JavaRDD<Vector> u = lectureFichier("/Users/Alain/Desktop/Data/net1_500_30.txt", jsc, nomAttributs);
	    
	    //Applying a function to extract extreme item sets
	    JavaRDD<String> v = (JavaRDD<String>) u.map(
		   new Function<Vector,String>() {
		     public String call(Vector v) {
		    	 StringBuilder s = new StringBuilder("");
		    	 for(int i=0; i< v.size(); i++) {
		    		 if (v.apply(i) < -0.5)
		    		 {
		    			 s.append(nomAttributs.get(i)+"_low ");
		    			 //s.append("A"+Integer.toString(i+1)+"_low ");
		    		 }
		    		 else if (v.apply(i) > 0.5)
		    		 {
		    			 s.append(nomAttributs.get(i)+"_high ");
		    			 //s.append("A"+Integer.toString(i+1)+"_high ");
		    		 }
		    	 }
		    	 return s.toString();
		     }
		   });
	    
	    //process before FPGrowth
	    JavaRDD<ArrayList<String>> transactions = v.map(
	  	      new Function<String, ArrayList<String>>() {
	  	        @Override
	  	        public ArrayList<String> call(String s) {
	  	          return Lists.newArrayList(s.split(" "));
	  	        }
	  	      }
	  	    );
	    
	    //creating FPGrowth model
	    FPGrowthModel<String> model = new FPGrowth()
	  	      .setMinSupport(minSupport)
	  	      .setNumPartitions(numPartition)
	  	      .run(transactions);
	    	    
	    //extracting association Rules from the model
	    AssociationRules arules = new AssociationRules().setMinConfidence(minConfidence);
	    JavaRDD<FreqItemset<String>> data = model.freqItemsets().toJavaRDD().cache();
	    JavaRDD<AssociationRules.Rule<String>> results = arules.run(data.cache());
		
	    double fin = (System.currentTimeMillis()-debut)/1000;
	    
	    //Display
	    System.out.println(u.take(5)); //take(n) display n first rows
	    System.out.println(v.take(5)); 
	    
//	    for (FPGrowth.FreqItemset<String> s: model.freqItemsets().toJavaRDD().collect()) {
//		      System.out.println("[" + Joiner.on(",").join(s.javaItems()) + "], " + s.freq());
//		    }
	   
	    
	    int nbExactRules = 0;
	    for (AssociationRules.Rule<String> rule : results.collect()) {
	    	if (rule.confidence() == 1)
	    		nbExactRules++;
		     // System.out.println(
		     // rule.javaAntecedent() + " => " + rule.javaConsequent() + ", " + rule.confidence());
		    }
	    System.out.println("NB atrributs : " + nomAttributs.size());
	    System.out.println("NB itemsets fréquents : " + data.count());
	    System.out.println("NB règles : " + results.cache().count());
	    System.out.println("NB règles exactes : " + nbExactRules);
	    System.out.println("Fini, temps de calcul : " + fin + "s");
	    //Spark closing
	    jsc.close();
	}
	
	public static JavaRDD<Vector> lectureFichier(String logFile, JavaSparkContext sc, List<String> nomAttributs)
	{				
		JavaRDD<String> lines = sc.textFile(logFile).cache();
		String premLigne = lines.first();
		nomAttributs.addAll(Arrays.asList(premLigne.split("\t")));
		lines = lines.filter(new 
				Function<String, Boolean>() {
			@Override
			public Boolean call(String arg0) throws Exception {
			return !premLigne.equals(arg0);
			}
			}); 
		JavaRDD<Vector> points = lines.map(new ParsePoint());		 
		
		// fonction "distinct()" pour retourner les lignes différentes du RDD.
		
		return points;
	}
	
	public static void main(String[] args) {
		try {
		double debut = System.currentTimeMillis();
		//associationRules();
		kMeans();
		double fin = (System.currentTimeMillis()-debut)/1000;
	    System.out.println("Fini, temps d'execution : " + fin + "s");
		} catch (Exception e) {
			System.err.println("Y'A UNE EXCEPTION !!" + e.getMessage());
		}
	}
}
