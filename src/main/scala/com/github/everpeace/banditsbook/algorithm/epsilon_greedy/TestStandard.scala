/*
 * Copyright (c) 2016 Shingo Omura
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

package com.github.everpeace.banditsbook.algorithm.epsilon_greedy

import java.io.{File, PrintWriter}

import breeze.linalg._
import breeze.stats.MeanAndVariance
import com.github.everpeace.banditsbook.arm._
import com.github.everpeace.banditsbook.testing_framework.TestRunner
import com.github.everpeace.banditsbook.testing_framework.TestRunner._
import com.typesafe.config.ConfigFactory

import scala.collection.immutable.Seq

object TestStandard extends _TestStandard with App {
  run()
}

trait _TestStandard {

  var baseKey = "banditsbook.algorithm.epsilon_greedy.test-standard"

  def runFor(configData: String) = {
    baseKey = configData
    run()
  }

  def run() = {
    //    implicit val randBasis = RandBasis.mt0

    val conf = ConfigFactory.load()

    val (_means, Some(εs), horizon, nSims, outDir) = readConfig(conf, baseKey, Some("εs"))
    val means = shuffle(_means)
    val arms = Seq(means: _*).map(μ => BernoulliArm(μ))


    var fileName = "test-standard-epsilon-greedy-results.csv"
    fileName = conf.getString(s"${baseKey}.output.file") match {
      case null => fileName
      case _ => conf.getString(s"${baseKey}.output.file")
    }
    val outputPath = new File(outDir, fileName)
    val file = new PrintWriter(outputPath.toString)
    file.write("epsilon, sim_num, step, chosen_arm, reward, cumulative_reward\n")
    try {
      println("---------------------------------")
      println("Standard Epsilon Greedy Algorithm")
      println("---------------------------------")
      println(s"   arms = ${means.map("(μ=" + _ + ")").mkString(", ")} (Best Arm = ${argmax(means)})")
      println(s"horizon = $horizon")
      println(s"  nSims = $nSims")
      println(s"      ε = (${εs.mkString(",")})")
      println("")

      val meanOfFinalRewards = scala.collection.mutable.Map.empty[Double, MeanAndVariance]
      val res = for {
        ε <- εs
      } yield {
        println(s"starts simulation on ε=$ε.")

        val algo = Standard.Algorithm(ε)
        val res = TestRunner.run(algo, arms, nSims, horizon)

        for {
          sim <- 0 until nSims
        } {
          val st = sim * horizon
          val end = ((sim + 1) * horizon) - 1
        }
        val finalRewards = res.cumRewards((horizon - 1) until(nSims * horizon, horizon))
        import breeze.stats._
        val meanAndVar = meanAndVariance(finalRewards)
        meanOfFinalRewards += ε -> meanAndVar
        println(s"reward stats: ${TestRunner.toString(meanAndVar)}")

        res.rawResults.valuesIterator.foreach { v =>
          file.write(s"${Seq(ε.toString, v._1.toString, v._2.toString, v._3.toString, v._4.toString, v._5.toString).mkString(",")}\n")
        }
        println(s"finished simulation on ε=$ε.")
      }
      println("")
      println(s"reward stats summary")
      println(s"${meanOfFinalRewards.iterator.toSeq.sortBy(_._1).map(p => (s"ε = ${p._1}", TestRunner.toString(p._2))).mkString("\n")}")
    } finally {
      file.close()
      println("")
      println(s"results are written to ${outputPath}")
    }
  }
}
