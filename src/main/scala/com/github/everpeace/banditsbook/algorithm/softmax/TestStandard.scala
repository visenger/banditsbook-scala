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

package com.github.everpeace.banditsbook.algorithm.softmax

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

  var baseKey = "banditsbook.algorithm.softmax.test-standard"

  def runFor(config: String): Unit = {
    baseKey = config
    run()
  }

  def run() = {
    //    implicit val randBasis = RandBasis.mt0

    val conf = ConfigFactory.load()

    val (_means, Some(τs), horizon, nSims, outDir) = readConfig(conf, baseKey, Some("τs"))
    val means = shuffle(_means)
    val arms = Seq(means: _*).map(μ => BernoulliArm(μ))

    var fileName = "test-standard-softmax-results.csv"
    fileName = conf.getString(s"${baseKey}.output.file") match {
      case null => fileName
      case _ => conf.getString(s"${baseKey}.output.file")
    }

    val outputPath = new File(outDir, fileName)
    val file = new PrintWriter(outputPath.toString)
    file.write("tau, sim_num, step, chosen_arm, reward, cumulative_reward\n")
    try {
      println("-------------------------------")
      println("Standard Softmax Algorithm")
      println("-------------------------------")
      println(s"   arms = ${means.map("(μ=" + _ + ")").mkString(", ")} (Best Arm = ${argmax(means)})")
      println(s"horizon = $horizon")
      println(s"  nSims = $nSims")
      println(s"      τ = (${τs.mkString(",")})")
      println("")

      val meanOfFinalRewards = scala.collection.mutable.Map.empty[Double, MeanAndVariance]
      val res = for {
        τ <- τs
      } yield {
        println(s"starts simulation on τ=$τ.")

        val algo = Standard.Algorithm(τ)
        val res: TestRunnerResult = TestRunner.run(algo, arms, nSims, horizon)

        for {
          sim <- 0 until nSims
        } {
          val st = sim * horizon
          val end = ((sim + 1) * horizon) - 1
        }
        val finalRewards = res.cumRewards((horizon - 1) until(nSims * horizon, horizon))
        import breeze.stats._
        val meanAndVar = meanAndVariance(finalRewards)
        meanOfFinalRewards += τ -> meanAndVar
        println(s"reward stats: ${TestRunner.toString(meanAndVar)}")

        res.rawResults.valuesIterator.foreach { v =>
          file.write(s"${Seq(τ, v._1, v._2, v._3, v._4, v._5).mkString(",")}\n")
        }
        println(s"finished simulation on τ=$τ.")
      }
      println("")
      println(s"reward stats summary")
      println(s"${meanOfFinalRewards.iterator.toSeq.sortBy(_._1).map(p => (s"τ=${p._1}", TestRunner.toString(p._2))).mkString("\n")}")
    } finally {
      file.close()
      println("")
      println(s"results are written to ${outputPath}")
    }
  }

  def runLog() = {
    //    implicit val randBasis = RandBasis.mt0

    val conf = ConfigFactory.load()

    val (_means, Some(τs), horizon, nSims, outDir) = readConfig(conf, baseKey, Some("τs"))
    val means = shuffle(_means)
    val arms = Seq(means: _*).map(μ => BernoulliArm(μ))

    var fileName = "test-standard-softmax-results.csv"
    fileName = conf.getString(s"${baseKey}.output.file") match {
      case null => fileName
      case _ => conf.getString(s"${baseKey}.output.file")
    }

    val outputPath = new File(outDir, fileName)
    val file = new PrintWriter(outputPath.toString)
    file.write("tau, sim_num, step, chosen_arm, reward, cumulative_reward\n")
    try {
      println("-------------------------------")
      println("Standard Softmax Algorithm")
      println("-------------------------------")
      println(s"   arms = ${means.map("(μ=" + _ + ")").mkString(", ")} (Best Arm = ${argmax(means)})")
      println(s"horizon = $horizon")
      println(s"  nSims = $nSims")
      println(s"      τ = (${τs.mkString(",")})")
      println("")

      val meanOfFinalRewards = scala.collection.mutable.Map.empty[Double, MeanAndVariance]
      val res = for {
        τ <- τs
      } yield {
        println(s"starts simulation on τ=$τ.")

        val algo = Standard.Algorithm(τ)
        val res: TestRunnerResult = TestRunner.run(algo, arms, nSims, horizon)

        for {
          sim <- 0 until nSims
        } {
          val st = sim * horizon
          val end = ((sim + 1) * horizon) - 1
        }
        val finalRewards = res.cumRewards((horizon - 1) until(nSims * horizon, horizon))
        import breeze.stats._
        val meanAndVar = meanAndVariance(finalRewards)
        meanOfFinalRewards += τ -> meanAndVar
        println(s"reward stats: ${TestRunner.toString(meanAndVar)}")

        res.rawResults.valuesIterator.foreach { v =>
          file.write(s"${Seq(τ, v._1, v._2, v._3, v._4, v._5).mkString(",")}\n")
        }
        println(s"finished simulation on τ=$τ.")
      }
      println("")
      println(s"reward stats summary")
      println(s"${meanOfFinalRewards.iterator.toSeq.sortBy(_._1).map(p => (s"τ=${p._1}", TestRunner.toString(p._2))).mkString("\n")}")
    } finally {
      file.close()
      println("")
      println(s"results are written to ${outputPath}")
    }
  }
}
