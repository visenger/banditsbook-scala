package com.github.everpeace.banditsbook

import java.io.{File, PrintWriter}

import breeze.linalg.Vector
import com.github.everpeace.banditsbook.algorithm.epsilon_greedy.Standard.State
import com.github.everpeace.banditsbook.arm.BernoulliArm
import com.typesafe.config.ConfigFactory

import scala.Seq
import scala.io.Source

/**
  * Created by visenger on 21/02/17.
  */

trait ExperimentsBase {


  val config = ConfigFactory.load()

  import scala.collection.convert.decorateAsScala._

  def means(baseKey: String) = Array(
    config.getDoubleList(s"$baseKey.arm-means").asScala.map(_.toDouble): _*
  )

  def bernoulliArms(means: Array[Double]) = {
    scala.collection.immutable.Seq(means: _*)
      .map(μ => BernoulliArm(μ))
  }

  def formatVector(expectations: Vector[Double]): String = {
    expectations.toDenseVector.activeIterator.map(e => {
      val idx = e._1 + 1
      s"$idx:${e._2}"
    }).mkString("|")


  }

  val epsilonGreedyConfig = "banditsbook.algorithm.epsilon_greedy"
  val softMaxConfig = "banditsbook.algorithm.softmax"
  val exp3Config = "banditsbook.algorithm.exp3"
  val ucbConfig = "banditsbook.algorithm.ucb.ucb1"

  val algConfigs = Seq(epsilonGreedyConfig, softMaxConfig, exp3Config, ucbConfig)

  val blackoak = "blackoak"
  val hosp = "hosp"
  val salaries = "salaries"

  val datasets = Seq(blackoak, hosp, salaries)

  val sep = ","
  val newLine = "\n"

  def initAlg(f: Double => Unit) = Seq(0.1d, 0.2d, 0.3d, 0.4d, 0.5d, 0.7d, 0.8d, 0.96d, 1.0d).foreach(i => f(i))

  def init_experiment(f: String => String): Seq[String] = {
    datasets.map(d => f(d))
  }

  def repeat(n: Int)(f: => Unit) = {
    0 to n foreach (i => f)
  }

  def write_to_file(path: String)(writer: PrintWriter => Unit) = {
    val file = new PrintWriter(new File(path))
    writer(file)
    file.close()
  }

}

object SoftMaxExperiments extends ExperimentsBase {

  def run() = {

    import algorithm._

    val softmaxExperiments = init_experiment {
      dataset => {
        val baseKey = s"$softMaxConfig.$dataset"

        val sims = config.getInt(s"$baseKey.n-sims")
        val τ = config.getDouble(s"$baseKey.best.τ")
        val ms: Array[Double] = means(baseKey)
        val banditArms = bernoulliArms(ms)

        val softMax = softmax.Standard.Algorithm(τ)
        var softMaxState = softMax.initialState(banditArms)
        repeat(sims) {
          val chosenArm: Int = softMax.selectArm(banditArms, softMaxState)
          val reward = banditArms(chosenArm).draw()
          softMaxState = softMax.updateState(banditArms, softMaxState, chosenArm, reward)
        }
        s"$dataset,softmax,${softMaxState.τ},${formatVector(softMaxState.expectations)}"
      }
    }
    softmaxExperiments.mkString(newLine)
  }


}

object EpsilonGreedyExperiments extends ExperimentsBase {
  def run(): String = {
    import algorithm._

    val epsilonGreedyExperiments = init_experiment {
      dataset => {
        val baseKey = s"$epsilonGreedyConfig.$dataset"

        val sims = config.getInt(s"$baseKey.n-sims")
        val ε = config.getDouble(s"$baseKey.best.ε")
        val ms: Array[Double] = means(baseKey)
        val banditArms = bernoulliArms(ms)

        val epsGreedy = epsilon_greedy.Standard.Algorithm(ε)
        var epsGreedyState: State = epsGreedy.initialState(banditArms)

        repeat(sims) {
          val chosenArm: Int = epsGreedy.selectArm(banditArms, epsGreedyState)
          val reward = banditArms(chosenArm).draw()
          epsGreedyState = epsGreedy.updateState(banditArms, epsGreedyState, chosenArm, reward)
        }
        s"$dataset,epsilon-greedy,${epsGreedyState.ε},${formatVector(epsGreedyState.expectations)}"
      }
    }

    epsilonGreedyExperiments.mkString(newLine)

  }
}

object Exp3Experiments extends ExperimentsBase {
  def run(): String = {

    import algorithm._

    val exp3Experiments = init_experiment {
      dataset => {
        val baseKey = s"$exp3Config.$dataset"

        val sims = config.getInt(s"$baseKey.n-sims")
        val γ = config.getDouble(s"$baseKey.best.γ")
        val ms: Array[Double] = means(baseKey)
        val banditArms = bernoulliArms(ms)

        val exp3alg = exp3.Exp3.Algorithm(γ)
        var exp3algState = exp3alg.initialState(banditArms)
        repeat(sims) {
          val chosenArm: Int = exp3alg.selectArm(banditArms, exp3algState)
          val reward = banditArms(chosenArm).draw()
          exp3algState = exp3alg.updateState(banditArms, exp3algState, chosenArm, reward)
        }
        s"$dataset,exp3,${exp3algState.γ},${formatVector(exp3algState.weights)}"
      }
    }
    exp3Experiments.mkString(newLine)
  }
}

object UCBExperiments extends ExperimentsBase {

  def run(): String = {

    import algorithm._

    val ucbExperiments = init_experiment {
      dataset => {
        val baseKey = s"$ucbConfig.$dataset"

        val sims = config.getInt(s"$baseKey.n-sims")
        val ms: Array[Double] = means(baseKey)
        val banditArms = bernoulliArms(ms)

        val ucbAlg = ucb.UCB1.Algorithm
        var ucbAlgState = ucbAlg.initialState(banditArms)
        repeat(sims) {
          val chosenArm: Int = ucbAlg.selectArm(banditArms, ucbAlgState)
          val reward = banditArms(chosenArm).draw()
          ucbAlgState = ucbAlg.updateState(banditArms, ucbAlgState, chosenArm, reward)
        }
        s"$dataset,ucb,,${formatVector(ucbAlgState.expectations)}"
      }
    }
    ucbExperiments.mkString(newLine)
  }
}

object AllExperimentsRunner extends ExperimentsBase {

  val experimentsHeader = config.getString("banditsbook.algorithm.experiments-common.header")
  val filePath = config.getString("banditsbook.algorithm.experiments-common.result.csv")

  def main(args: Array[String]): Unit = {
    val epsilonGreedyExperiments = EpsilonGreedyExperiments.run()
    val softMaxExperiments = SoftMaxExperiments.run()
    val exp3Experiments = Exp3Experiments.run()
    val ucbExperiments = UCBExperiments.run()

    write_to_file(filePath) {
      file => {

        file.write(s"$experimentsHeader$newLine")

        file.write(s"$epsilonGreedyExperiments$newLine")

        file.write(s"$softMaxExperiments$newLine")

        file.write(s"$exp3Experiments$newLine")

        file.write(s"$ucbExperiments$newLine")
      }
        println(s"done: see file $filePath")
    }


  }
}



