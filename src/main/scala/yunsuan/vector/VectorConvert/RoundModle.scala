package yunsuan.vector.VectorConvert

import chisel3._
import chisel3.util._
import yunsuan.vector.VectorConvert.RoundingModle._

class RoundingUnit(val width: Int) extends Module {
  val io = IO(new Bundle() {
    val in = Input(UInt(width.W))
    val roundIn = Input(Bool())
    val stickyIn = Input(Bool())
    val signIn = Input(Bool())
    val rm = Input(UInt(3.W))
    val out = Output(UInt(width.W))
    val inexact = Output(Bool())
    val cout = Output(Bool())
    val r_up = Output(Bool())
  })

  val (g, r, s) = (io.in(0).asBool(), io.roundIn, io.stickyIn)
  val inexact = r | s
  val r_up = MuxLookup( //舍入后绝对值增加
    io.rm,
    false.B,
    Seq(
      RNE -> ((r && s) || (r && !s && g)),
      RTZ -> false.B,
      RUP -> (inexact & !io.signIn),
      RDN -> (inexact & io.signIn),
      RMM -> r,
      RTO -> ((r || s) && !g),
    )
  )
  val out_r_up = io.in + 1.U  //尝试把这玩意儿移出去
  io.out := Mux(r_up, out_r_up, io.in) //如果有进位，这里变全零 float格式的隐含1还在，并且在外面exp + 1, todo: 如果是整数呢??
  io.inexact := inexact
  // r_up && io.in === 111...1
  io.cout := r_up && io.in.andR()
  io.r_up := r_up
}

//class RoundingUnitNew(val width: Int) extends Module {
//  val io = IO(new Bundle() {
//    val in = Input(UInt(width.W))
//    val roundIn = Input(Bool())
//    val stickyIn = Input(Bool())
//    val signIn = Input(Bool())
//    val rm = Input(UInt(3.W))
//
//    val out = Output(UInt(width.W))
//    val inexact = Output(Bool())
//    val cout = Output(Bool())
//    val r_up = Output(Bool())
//  })
//
//  val (g, r, s) = (io.in(0).asBool(), io.roundIn, io.stickyIn)
//  val inexact = r | s
//  val r_up = MuxLookup( //舍入后绝对值增加
//    io.rm,
//    false.B,
//    Seq(
//      RNE -> ((r && s) || (r && !s && g)),
//      RTZ -> false.B,
//      RUP -> (inexact & !io.signIn),
//      RDN -> (inexact & io.signIn),
//      RMM -> r,
//      RTO -> ((r && s) || (r && !s && !g))
//    )
//  )
//
//  io.out := io.in
//  io.inexact := inexact
//  io.cout := r_up && io.in.andR()
//  io.r_up := r_up
//
////  val out_r_up = io.in + 1.U  //尝试把这玩意儿移出去
////  io.out := Mux(r_up, out_r_up, io.in) //如果有进位，这里变全零 float格式的隐含1还在，并且在外面exp + 1, todo: 如果是整数呢??
////  io.inexact := inexact
////  // r_up && io.in === 111...1
////  io.cout := r_up && io.in.andR()
////  io.r_up := r_up
//}


object RoundingUnit {
  def apply(in: UInt, rm: UInt, sign: Bool, width: Int): RoundingUnit = {
    require(in.getWidth >= width)
    val in_pad = if(in.getWidth < width + 2) padd_tail(in, width + 2) else in
    val rounder = Module(new RoundingUnit(width))
    rounder.io.in := in_pad.head(width)
    rounder.io.roundIn := in_pad.tail(width).head(1).asBool()
    rounder.io.stickyIn := in_pad.tail(width + 1).orR()
    rounder.io.rm := rm
    rounder.io.signIn := sign
    rounder
  }
  def padd_tail(x: UInt, w: Int): UInt = Cat(x, 0.U((w - x.getWidth).W)) // 少于两位则补零
  def is_rmin(rm: UInt, sign: Bool): Bool = {
    rm === RTZ || (rm === RDN && !sign) || (rm === RUP && sign)
  }
}

//class TininessRounder(expWidth: Int, precision: Int) extends Module {
//
//  val io = IO(new Bundle() {
//    val in = Input(new RawFloat(expWidth, precision + 3))
//    val rm = Input(UInt(3.W))
//    val tininess = Output(Bool())
//  })
//
//  val rounder = RoundingUnit(
//    io.in.sig.tail(2),
//    io.rm,
//    io.in.sign,
//    precision - 1
//  )
//
//  val tininess = io.in.sig.head(2) === "b00".U(2.W) ||
//    (io.in.sig.head(2) === "b01".U(2.W) && !rounder.io.cout)
//
//  io.tininess := tininess
//}