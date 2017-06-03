name := "spark-optim"

version := "1.0"

scalaVersion := "2.11.8"

spName := "hibayesian/spark-optim"

sparkVersion := "2.1.1"

sparkComponents += "mllib"

resolvers += Resolver.sonatypeRepo("public")

spShortDescription := "spark-optim"

spDescription := """A library of scalable optimization algorithms based on Spark"""
  .stripMargin

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")
    