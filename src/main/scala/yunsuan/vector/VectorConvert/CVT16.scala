package yunsuan.vector.VectorConvert

import chisel3._
import chisel3.util._
import yunsuan.util.LookupTree
import yunsuan.vector.VectorConvert.util.{CLZ, ShiftRightJam, VFRSqrtTable, VFRecTable, RoundingUnit}
import yunsuan.vector.VectorConvert.RoundingModle._


class CVT16(width: Int = 16) extends CVT(width){

  val is_single = io.opType.tail(3).head(2) === "b00".U
  val is_widen  = io.opType.tail(3).head(2) === "b01".U
  val is_narrow = io.opType.tail(3).head(2) === "b10".U
  val is_single_reg0 = RegNext(is_single, false.B)
  val is_widen_reg0 = RegNext(is_widen, false.B)
  val is_narrow_reg0 = RegNext(is_narrow, false.B)

  val is_vfr = io.opType(5).asBool
  val is_fp2int = io.opType.head(2) === "b10".U && !is_widen
  val is_int2fp = io.opType.head(2) === "b01".U && !is_narrow

  val is_signed_int = io.opType(0)
  val is_signed_int_reg0 = RegNext(is_signed_int)

  val round_in = Wire(UInt(11.W))
  val round_in_reg0 = RegNext(round_in, 0.U(11.W))
  val round_roundIn = Wire(Bool())
  val round_roundIn_reg0 = RegNext(round_roundIn, false.B)
  val round_stickyIn = Wire(Bool())
  val round_stickyIn_reg0 = RegNext(round_stickyIn, false.B)
  val round_signIn = Wire(Bool())
  val round_signIn_reg0 = RegNext(round_signIn, false.B)
  val rm_reg0 = RegNext(io.rm)


  val result = Wire(UInt(16.W))
  val NV, DZ, OF, UF, NX = WireInit(false.B)
  val fflags = WireInit(Cat(NV, DZ, OF, UF, NX))

  val result0 = Wire(UInt(16.W))
  val result0_reg0 = RegNext(result0, 0.U(16.W))
  val fflags0 = WireInit(Cat(NV, DZ, OF, UF, NX))
  val fflags0_reg0 = RegNext(fflags0)


  /**
   * fp16 -> ui16  vfcvt.xu.f.v  vfcvt.rtz.xu.f.v
   *      -> i16   vfcvt.x.f.v   vfcvt.rtz.x.f.v
   *      -> ui8   vfncvt.xu.f.w vfncvt.rtz.xu.f.w
   *      -> i8    vfncvt.x.f.w  vfncvt.rtz.x.f.w
   */
  when(is_fp2int) {
    val in = VectorFloat.fromUInt(io.src, f16.expWidth, f16.precision)
    val raw_in = RawVectorFloat.fromVFP(in, Some(in.decode.expNotZero))
    val raw_in_reg0 = RegNext(raw_in)
    
    // single max exp = bias +& 15.U, narrow max exp = bias +& 7.U
    val max_int_exp = Mux(is_single, 30.U, 22.U)
    val exp_of = raw_in.exp > max_int_exp

    // left
    // only f16->i16(ui16) can left shift, shamt is raw_in.exp - (bias + precision - 1)
    val lpath_shamt = Mux(is_single, raw_in.exp - 25.U, 0.U)
    // max shamt width = (15 - (precision - 1)).U.getWidth
    val lpath_sig_shifted = Mux(is_single, (raw_in.sig << lpath_shamt(4, 0))(15, 0), 0.U)
    // fp->ui16, if fp is negative, invalid
    val lpath_iv = is_single && !is_signed_int && raw_in.sign
    // f16->i16, may overflow
    val lpath_may_of = is_single && is_signed_int && (raw_in.exp === max_int_exp)
    val lpath_pos_of = is_single && lpath_may_of && !raw_in.sign
    val lpath_neg_of = is_single && lpath_may_of && raw_in.sign && raw_in.sig.tail(1).orR
    val lpath_of = lpath_pos_of || lpath_neg_of

    // right
    // f16->i8(ui8) always right shift
    val rpath_shamt = Mux(is_single, 25.U - raw_in.exp, 22.U - raw_in.exp);
    val (rpath_sig_shifted, rpath_sticky) = ShiftRightJam(Cat(raw_in.sig, 0.U), rpath_shamt)
    val rpath_rounder = Module(new RoundingUnit(f16.precision))
    round_in := Mux(is_single, rpath_sig_shifted.head(f16.precision), rpath_sig_shifted.head(8))
    round_roundIn := Mux(is_single, rpath_sig_shifted.tail(f16.precision), rpath_sig_shifted.tail(8).head(1))
    round_stickyIn := rpath_sticky | Mux(is_narrow, rpath_sig_shifted.tail(9).orR, false.B)
    round_signIn := raw_in.sign
    rpath_rounder.io.in := round_in_reg0
    rpath_rounder.io.roundIn := round_roundIn_reg0
    rpath_rounder.io.stickyIn := round_stickyIn_reg0
    rpath_rounder.io.signIn := round_signIn_reg0
    rpath_rounder.io.rm := rm_reg0

    val rpath_out_reg0 = Mux(rpath_rounder.io.r_up, rpath_rounder.io.in + 1.U, rpath_rounder.io.in)
    val rpath_cout_reg0 = rpath_rounder.io.r_up && Mux(is_narrow_reg0, rpath_rounder.io.in.tail(4).andR, rpath_rounder.io.in.andR)

    val rpath_sig_reg0 = Mux(is_single_reg0, Cat(0.U(4.W), rpath_cout_reg0, rpath_out_reg0), Cat(0.U(5.W), rpath_out_reg0))
    val rpath_ix_reg0 = rpath_rounder.io.inexact || is_narrow_reg0 && RegNext(rpath_sig_shifted.tail(8).orR)
    val rpath_iv_reg0 = !is_signed_int_reg0 && raw_in_reg0.sign && rpath_sig_reg0.orR
    val rpath_pos_of_reg0 = !raw_in_reg0.sign &&
      Mux(is_signed_int,
        (raw_in_reg0.exp === 22.U) || ((raw_in_reg0.exp === 21.U) && rpath_cout_reg0),
        (raw_in_reg0.exp === 22.U) && rpath_cout_reg0)
    val rpath_neg_of_reg0 = raw_in_reg0.sign && (raw_in_reg0.exp === 22.U) && (rpath_rounder.io.in.tail(4).orR || rpath_rounder.io.r_up)
    val rpath_of_reg0 = Mux(is_narrow_reg0, rpath_pos_of_reg0 || rpath_neg_of_reg0, rpath_cout_reg0)

    // select result
    val sel_lpath = raw_in.exp >= 25.U
    val sel_lpath_reg0 = RegNext(sel_lpath)
    val of = RegNext(exp_of) || sel_lpath_reg0 && RegNext(lpath_of) || !sel_lpath_reg0 && rpath_of_reg0
    val iv = of || sel_lpath_reg0 && RegNext(lpath_iv) || !sel_lpath_reg0 && rpath_iv_reg0
    val ix = !iv && !sel_lpath_reg0 && rpath_ix_reg0

    val int_abs = Mux(sel_lpath_reg0, RegNext(lpath_sig_shifted), rpath_sig_reg0)
    val int = Mux(is_narrow_reg0,
      Mux(raw_in_reg0.sign && is_signed_int_reg0, -int_abs.tail(8), int_abs.tail(8)),
      Mux(raw_in_reg0.sign && is_signed_int_reg0, -int_abs, int_abs))

    val max_int = Mux(is_single, Cat(!is_signed_int, ~0.U(15.W)), Cat(!is_signed_int, ~0.U(7.W)))
    val min_int = Mux(is_single, Cat(is_signed_int,   0.U(15.W)), Cat(is_signed_int,   0.U(7.W)))

    result0 := Mux(iv, RegNext(Mux(in.decode.isNaN | !raw_in.sign, max_int, min_int)), int)
    fflags0 := Cat(iv, false.B, false.B, false.B, ix)

    result := result0_reg0
    fflags := fflags0_reg0
  }.elsewhen(is_int2fp) {
    /**
     * ui16 -> f16  vfcvt.f.xu.v
     * i16  ->      vfcvt.f.x.v
     * ui8  ->      vfwcvt.f.xu.v
     * i8   ->      vfwcvt.f.x.v
     */
    val sign = is_signed_int && Mux(is_widen, io.src(7), io.src(15))
    val in_sext = Cat(Fill(8, io.src(7)), io.src(7,0))
    val in = Mux(is_signed_int && is_widen, in_sext, io.src)
    val in_abs = Mux(sign, (~in).asUInt + 1.U, in)

    val lzc = CLZ(in_abs)
    val in_shift = (in_abs << lzc)(14, 0)
    val exp_raw = 30.U - lzc

    val rounder = Module(new RoundingUnit(10))
    round_in := Mux(is_widen, in_shift.head(8), in_shift.head(10))
    round_roundIn := Mux(is_widen, in_shift.tail(8).head(1), in_shift.tail(10).head(1)).asBool
    round_stickyIn := Mux(is_widen, in_shift.tail(9).orR, in_shift.tail(f16.precision).orR)
    round_signIn := sign
    rounder.io.in := round_in_reg0
    rounder.io.roundIn := round_roundIn_reg0
    rounder.io.stickyIn := round_stickyIn_reg0
    rounder.io.signIn := round_signIn_reg0
    rounder.io.rm := rm_reg0

    val out_reg0 = Mux(rounder.io.r_up, rounder.io.in + 1.U, rounder.io.in)
    val cout_reg0 = rounder.io.r_up && rounder.io.in.andR.asBool

    val exp_reg0 = Mux(in.orR, RegNext(exp_raw) + cout_reg0, 0.U)
    val sig_reg0 = out_reg0

    val of = exp_reg0 === 31.U
    val ix = rounder.io.inexact

    result0 := Cat(is_signed_int_reg0 && RegNext(sign), exp_reg0, Mux(is_widen_reg0, Cat(sig_reg0.tail(2), 0.U(2.W)), sig_reg0))
    fflags0 := Cat(false.B, false.B, of, false.B, ix)

    result := result0_reg0
    fflags := fflags0_reg0
  }.otherwise {
    round_in := 0.U
    round_roundIn := false.B
    val is_vfrsqrt7 = !io.opType(0).asBool
    val is_vfrsqrt7_reg0 = RegNext(is_vfrsqrt7)
    val is_vfrec7 = io.opType(0).asBool
    val is_vfrec7_reg0 = RegNext(is_vfrec7)
    val vfrsqrt7Table = Module(new Rsqrt7Table)
    val vfrec7Table = Module(new Rec7Table)

    val in = io.src
    val sign = in.head(1).asBool
    val sign_reg0 = RegNext(sign)
    val exp = in.tail(1).head(f16.expWidth)
    val sig = in.tail(6)

    val is_normal = exp.orR & !exp.andR
    val is_normal_reg0 = RegNext(is_normal)
    val is_subnormal = !exp.orR
    val is_inf = exp.andR & !sig.orR
    val is_inf_reg0 = RegNext(is_inf)
    val is_nan = exp.andR & sig.orR
    val is_nan_reg0 = RegNext(is_nan)
    val is_neginf = sign & is_inf
    val is_neginf_reg0 = RegNext(is_neginf)
    val is_neginf_negzero = sign & (is_normal | is_subnormal & sig.orR)
    val is_neginf_negzero_reg0 = RegNext(is_neginf_negzero)
    val is_negzero = sign & is_subnormal & !sig.orR
    val is_negzero_reg0 = RegNext(is_negzero)
    val is_poszero = !sign & is_subnormal & !sig.orR
    val is_poszero_reg0 = RegNext(is_poszero)
    val is_poszero_posinf = !sign & (is_normal | is_subnormal & sig.orR)
    val is_snan = !sig.head(1).asBool & is_nan
    val is_snan_reg0 = RegNext(is_snan)
    val is_neg2_bplus1_b = sign & (exp === 30.U)
    val is_neg2_bplus1_b_reg0 = RegNext(is_neg2_bplus1_b)
    val is_neg2_b_bminus1 = sign & (exp === 29.U)
    val is_neg2_b_bminus1_reg0 = RegNext(is_neg2_b_bminus1)
    val is_neg2_negbminus1_negzero = sign & (sig.head(2) === "b00".U) & is_subnormal & sig.orR
    val is_neg2_negbminus1_negzero_reg0 = RegNext(is_neg2_negbminus1_negzero)
    val is_pos2_poszero_negbminus1 = !sign & (sig.head(2) === "b00".U) & is_subnormal & sig.orR
    val is_pos2_poszero_negbminus1_reg0 = RegNext(is_pos2_poszero_negbminus1)
    val is_pos2_bminus1_b = !sign & (exp === 29.U)
    val is_pos2_bminus1_b_reg0 = RegNext(is_pos2_bminus1_b)
    val is_pos2_b_bplus1 = !sign & (exp === 30.U)
    val is_pos2_b_bplus1_reg0 = RegNext(is_pos2_b_bplus1)


     val zero_minus_lzc = 0.U - CLZ(sig) // 0 - count leading zero
     val exp_normalized =
       Mux(is_vfrsqrt7,
         Mux(is_poszero_posinf, Mux(is_normal, exp, Cat(Fill(f16.expWidth - zero_minus_lzc.getWidth, zero_minus_lzc.head(1)), zero_minus_lzc)), 0.U),
         Mux(is_normal, exp, Cat(Fill(f16.expWidth - zero_minus_lzc.getWidth, zero_minus_lzc.head(1)), zero_minus_lzc)))

     val exp_normalized_reg0 = RegNext(exp_normalized)

     val sig_normalized = Wire(UInt(11.W))
     sig_normalized := Mux(is_vfrsqrt7, Mux(is_poszero_posinf, Mux(is_normal, Cat(0.U, sig), (sig << 1.U).asUInt), 0.U), Mux(is_normal, Cat(0.U, sig), (sig << 1.U).asUInt))

     val sig_normalized_reg0 = RegNext(sig_normalized)

    val clz_reg0 = RegNext(CLZ(sig_normalized))

     val sig_in7 = Mux(is_vfrsqrt7_reg0,
       Cat(exp_normalized_reg0(0), (sig_normalized_reg0 << Mux(is_normal_reg0, 0.U, clz_reg0))(9, 4)), // vfrsqrt7  Cat(exp_nor(0), sig_nor(9,4))
       (sig_normalized_reg0 << Mux(is_normal_reg0, 0.U, clz_reg0))(9,3)) // vfrec7 sig_nor(9,3)
    
    vfrsqrt7Table.src := sig_in7
    vfrec7Table.src := sig_in7

    val sig_out7 = Wire(UInt(7.W))
    sig_out7 := Mux(is_vfrsqrt7_reg0, vfrsqrt7Table.out, vfrec7Table.out)

    val out_exp_normalized = Mux(is_vfrec7, 29.U - exp_normalized, 0.U) // 2 * bias - 1 - exp_nor
    val out_exp_is_zero_negone = !out_exp_normalized.orR || out_exp_normalized.andR
    val out_exp_is_zero_negone_reg0 = RegNext(out_exp_is_zero_negone)
    val out_exp = Wire(UInt(5.W))
    out_exp := Mux(is_vfrsqrt7,
      Mux(is_normal, (44.U - exp) >> 1, (44.U + CLZ(sig)) >> 1), // if normal (3 * bias - 1 - exp) >> 1 else (3 * bias -1 + CLZ) >>1
      Mux(out_exp_is_zero_negone, 0.U, out_exp_normalized))
    val out_exp_reg0 = RegNext(out_exp)
    val out_sig =
      Mux(is_vfrec7_reg0,
        Mux(out_exp_is_zero_negone_reg0,
          Mux(is_neg2_bplus1_b_reg0 || is_pos2_b_bplus1_reg0,
            Cat(0.U, 1.U, sig_out7, 0.U),
            Mux(is_neg2_b_bminus1_reg0 || is_pos2_bminus1_b_reg0,
              Cat(1.U, sig_out7, 0.U(2.W)),
              Cat(1.U, sig_out7, 0.U(2.W)) >> 1.U)),
          Cat(sig_out7, 0.U(3.W))),
        0.U)

    val out_sign = is_poszero_posinf & sign
    val out_sign_reg0 = RegNext(out_sign)

    val fp_result = Wire(UInt(16.W))
    fp_result := Mux1H(
      Seq(is_vfrsqrt7_reg0,
        is_vfrec7_reg0),
      Seq(Cat(out_sign_reg0, out_exp_reg0, sig_out7, 0.U(3.W)),
        Cat(sign_reg0, out_exp_reg0, out_sig))
    )

    val result_nan = Cat(0.U(1.W), Fill(6, 1.U), 0.U(9.W))
    val result_inf = Cat(Fill(5, 1.U), 0.U(10.W))
    val result_greatest_fin = Cat(Fill(4, 1.U), 0.U, Fill(10, 1.U))

    result0 := Mux1H(
      Seq(is_vfrsqrt7_reg0 && (is_nan_reg0 || is_neginf_negzero_reg0) || is_vfrec7_reg0 && is_nan_reg0,
        is_vfrsqrt7_reg0 && is_inf_reg0,
        is_vfrsqrt7_reg0 && (is_negzero_reg0 || is_poszero_reg0),
        is_vfrsqrt7_reg0 && !(is_nan_reg0 || is_neginf_negzero_reg0 || is_inf_reg0 || is_negzero_reg0 || is_poszero_reg0) ||
          is_vfrec7_reg0 && !(is_nan_reg0 || is_inf_reg0 || is_negzero_reg0 || is_poszero_reg0 || is_neg2_negbminus1_negzero_reg0 || is_pos2_poszero_negbminus1_reg0),
        is_vfrec7_reg0 && is_inf_reg0,
        is_vfrec7_reg0 && (is_negzero_reg0 || is_poszero_reg0),
        is_vfrec7_reg0 && is_neg2_negbminus1_negzero_reg0,
        is_vfrec7_reg0 && is_pos2_poszero_negbminus1_reg0
      ),
      Seq(result_nan,
        Mux(is_neginf_reg0, result_nan, 0.U),
        Mux(is_negzero_reg0, Cat(1.U, result_inf), Cat(0.U, result_inf)),
        fp_result,
        Mux(is_neginf_reg0, Cat(1.U, 0.U(15.W)), 0.U),
        Mux(is_negzero_reg0, Cat(Fill(6, 1.U), 0.U(10.W)), Cat(0.U(1.W), Fill(5, 1.U), 0.U(10.W))),
        Mux(rm_reg0 === RUP || rm_reg0 === RTZ, Cat(1.U, result_greatest_fin), Cat(1.U, result_inf)),
        Mux(rm_reg0 === RDN || rm_reg0 === RTZ, Cat(0.U, result_greatest_fin), Cat(0.U, result_inf))
      )
    )
    fflags0 := Mux1H(
      Seq(is_vfrsqrt7_reg0 && (is_nan_reg0 || is_neginf_negzero_reg0),
        is_vfrsqrt7_reg0 && is_inf_reg0,
        is_vfrsqrt7_reg0 && (is_negzero_reg0 || is_poszero_reg0) || is_vfrec7_reg0 && (is_negzero_reg0 || is_poszero_reg0),
        is_vfrsqrt7_reg0 && !(is_nan_reg0 || is_neginf_negzero_reg0 || is_inf_reg0 || is_negzero_reg0 || is_poszero_reg0) ||
          is_vfrec7_reg0 && !(is_nan_reg0 || is_inf_reg0 || is_negzero_reg0 || is_poszero_reg0 || is_neg2_negbminus1_negzero_reg0 || is_pos2_poszero_negbminus1_reg0),
        is_vfrec7_reg0 && is_nan_reg0,
        is_vfrec7_reg0 && is_inf_reg0,
        is_vfrec7_reg0 && (is_neg2_negbminus1_negzero_reg0 || is_pos2_poszero_negbminus1_reg0)
      ),
      Seq(Mux(is_snan_reg0 || is_neginf_negzero_reg0, "b10000".U, "b00000".U),
        Mux(is_neginf_reg0, "b10000".U, "b00000".U),
        "b01000".U,
        0.U,
        Mux(is_snan_reg0, "b10000".U, "b00000".U),
        Mux(is_neginf_reg0, Cat(1.U, 0.U(15.W)), 0.U),
        "b00101".U
      )
    )
    result := result0_reg0
    fflags := fflags0_reg0
   }

   io.result := result
   io.fflags := fflags
}
