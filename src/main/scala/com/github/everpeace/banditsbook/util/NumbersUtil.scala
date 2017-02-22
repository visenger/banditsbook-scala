package com.github.everpeace.banditsbook.util

import scala.math.BigDecimal.RoundingMode

/**
  * Util class for numbers.
  */
object NumbersUtil {

  def round(percentageFound: Double, scale: Int = 2) = {
    val number = percentageFound.isNaN match {
      case true => 0.0
      case false => percentageFound
    }
    BigDecimal(number).setScale(scale, RoundingMode.HALF_UP).toDouble
  }

}
