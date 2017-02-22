package com.github.everpeace.banditsbook

import com.github.everpeace.banditsbook.algorithm.epsilon_greedy.Standard.State
import com.github.everpeace.banditsbook.arm.BernoulliArm
import com.typesafe.config.ConfigFactory

/**
  * Created by visenger on 21/02/17.
  */

trait ExperimentsBase {

  val config = ConfigFactory.load()

  import scala.collection.convert.decorateAsScala._

  def means(baseKey: String) = Array(
    config.getDoubleList(s"$baseKey.arm-means").asScala.map(_.toDouble): _*
  )

  def bernoulliArms(means: Array[Double]) = scala.collection.immutable.Seq(means: _*).map(μ => BernoulliArm(μ))

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

  def initAlg(f: Double => Unit) = Seq(0.1d, 0.2d, 0.3d, 0.4d, 0.5d, 0.7d, 0.8d, 0.96d, 1.0d).foreach(i => f(i))

  def init_experiment(f: String => Unit) = {
    datasets.foreach(d => f(d))
  }

  def repeat(n: Int)(f: => Unit) = {
    0 to n foreach (i => f)
  }

}

object SoftMaxExperiments extends ExperimentsBase {

  def run() = {

    import algorithm._

    init_experiment {
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

        println(s"        SoftMax for ${dataset} ")
        println(s"        param: τ=${softMaxState.τ} ")
        println(s"       counts: [${softMaxState.counts.valuesIterator.mkString(sep)}]")
        println(s" expectations: [${softMaxState.expectations.valuesIterator.mkString(sep)}]")
      }
    }
  }


}

object EpsilonGreedyExperiments extends ExperimentsBase {
  def run() = {
    import algorithm._

    init_experiment {
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

        println(s"        EpsilonGreedy for ${dataset} ")
        println(s"        param: ε=${epsGreedyState.ε} ")
        println(s"       counts: [${epsGreedyState.counts.valuesIterator.mkString(sep)}]")
        println(s" expectations: [${epsGreedyState.expectations.valuesIterator.mkString(sep)}]")
      }
    }

  }
}

object Exp3Experiments extends ExperimentsBase {
  def run() = {

    import algorithm._

    init_experiment {
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

        println(s"        EXP3 for ${dataset} ")
        println(s"        param: γ=${exp3algState.γ} ")
        println(s"       counts: [${exp3algState.counts.valuesIterator.mkString(sep)}]")
        println(s"      weights: [${exp3algState.weights.valuesIterator.mkString(sep)}]")
      }
    }
  }
}

object UCBExperiments extends ExperimentsBase {

  def run() = {

    import algorithm._

    init_experiment {
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

        println(s"        UCB for ${dataset} ")
        println(s"       counts: [${ucbAlgState.counts.valuesIterator.mkString(sep)}]")
        println(s"      weights: [${ucbAlgState.expectations.valuesIterator.mkString(sep)}]")
      }
    }
  }
}

object AllExperimentsRunner {

  def main(args: Array[String]): Unit = {
    EpsilonGreedyExperiments.run()
    println("########################################################################")
    SoftMaxExperiments.run()
    println("########################################################################")
    Exp3Experiments.run()
    println("########################################################################")
    UCBExperiments.run()
  }
}



