// wordCount
import org.apache.spark.{SparkConf, SparkContext}

object WordCount {
  def main(args: Array[String]): Unit = {
    // 创建 SparkConf 对象，指定应用程序名称
    val conf = new SparkConf().setAppName("WordCount")
    conf.setMaster("local")
    // 创建 SparkContext 对象，用于与 Spark 集群通信
    val sc = new SparkContext(conf)
    // 读取文本文件并将每一行的文本字母小写，以空格分割并以列表的形式保存
    val lines = sc.textFile("hdfs://localhost:9000/usr/Input/test.txt").flatMap(_.toLowerCase.split(" "))
    // 对单词进行计数
    // 将单词转换为(word, 1)的形式并分组
    val wordGroups = lines.map((_,1)).groupBy(_._1)
    // 计算词频
    val wordCounts = wordGroups.map(x => (x._1, x._2.size))
    // 按照词频降序排序
    val sortedWordCounts = wordCounts.sortBy(_._2, ascending = false)
    // 使用 for 循环打印结果
    val collectedWords = sortedWordCounts.collect()
    for ((word, count) <- collectedWords) {
      println(s"$word: $count")
    }
  }
}