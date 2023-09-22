package yunsuan.vector.VectorConvert

import chisel3._
import chisel3.util._
import yunsuan.util.LookupTree
import yunsuan.vector.VectorConvert.util.{CLZ, ShiftRightJam, VFRSqrtTable, VFRecTable, RoundingUnit}
import yunsuan.vector.VectorConvert.RoundingModle._


class CVT16(width: Int = 16) extends CVT(width){

/*
  val is_single = io.opType.tail(3).head(2) === "b00".U
  val is_widen  = io.opType.tail(3).head(2) === "b01".U
  val is_narrow = io.opType.tail(3).head(2) === "b10".U
  val is_single_reg0 = RegNext(is_single, false.B)
  val is_widen_reg0 = RegNext(is_widen, false.B)
  val is_narrow_reg0 = RegNext(is_narrow, false.B)

  val is_vfr = io.opType(5).asBool
  val is_fp2int = io.opType.head(2) === "b10".U && !is_widen
  val is_int2fp = io.opType.head(2) === "b01".U && !is_narrow
  val is_vfr_reg0 = RegNext(is_vfr)
  val is_fp2int_reg0 = RegNext(is_fp2int)
  val is_int2fp_reg0 = RegNext(is_int2fp)

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


  // fp2int
  val raw_in = Wire(UInt(17.W))
  val raw_in_reg0 = RegNext(raw_in, 0.U(17.W))

  val rpath_sig_shifted0 = Wire(UInt(12.W))
  val rpath_sig_shifted_reg0 = RegNext(rpath_sig_shifted0)
  val sel_lpath = Wire(Bool())
  val sel_lpath_reg0 = RegNext(sel_lpath)
  val exp_of = Wire(Bool())
  val exp_of_reg0 = RegNext(exp_of)
  val lpath_of = Wire(Bool())
  val lpath_of_reg0 = RegNext(lpath_of)
  val lpath_iv = Wire(Bool())
  val lpath_iv_reg0 = RegNext(lpath_iv)
  val lpath_sig_shifted = Wire(UInt(16.W))
  val lpath_sig_shifted_reg0 = RegNext(lpath_sig_shifted)
  val max_min_int = Wire(UInt(16.W))
  val max_min_int_reg0 = RegNext(max_min_int)

  // int2fp
  val in_orR = Wire(Bool())
  val in_orR_reg0 = RegNext(in_orR)
  val exp_raw = Wire(UInt(5.W))
  val exp_raw_reg0= RegNext(exp_raw)
  val sign = Wire(Bool())
  val sign_reg0 = RegNext(sign)

  // vfr
  val is_vfrsqrt7 = is_vfr && !io.opType(0).asBool
  val is_vfrec7 = is_vfr && io.opType(0).asBool
  val is_vfrsqrt7_reg0 = RegNext(is_vfrsqrt7)
  val is_vfrec7_reg0 = RegNext(is_vfrec7)
  val is_normal = Wire(Bool())
  val is_inf = Wire(Bool())
  val is_nan = Wire(Bool())
  val is_neginf = Wire(Bool())
  val is_neginf_negzero = Wire(Bool())
  val is_negzero = Wire(Bool())
  val is_poszero = Wire(Bool())
  val is_snan = Wire(Bool())
  val is_neg2_bplus1_b = Wire(Bool())
  val is_neg2_b_bminus1 = Wire(Bool())
  val is_neg2_negbminus1_negzero = Wire(Bool())
  val is_pos2_poszero_negbminus1 = Wire(Bool())
  val is_pos2_bminus1_b = Wire(Bool())
  val is_pos2_b_bplus1 = Wire(Bool())

  val is_normal_reg0 = RegNext(is_normal)
  val is_inf_reg0 = RegNext(is_inf)
  val is_nan_reg0 = RegNext(is_nan)
  val is_neginf_reg0 = RegNext(is_neginf)
  val is_neginf_negzero_reg0 = RegNext(is_neginf_negzero)
  val is_negzero_reg0 = RegNext(is_negzero)
  val is_poszero_reg0 = RegNext(is_poszero)
  val is_snan_reg0 = RegNext(is_snan)
  val is_neg2_bplus1_b_reg0 = RegNext(is_neg2_bplus1_b)
  val is_neg2_b_bminus1_reg0 = RegNext(is_neg2_b_bminus1)
  val is_neg2_negbminus1_negzero_reg0 = RegNext(is_neg2_negbminus1_negzero)
  val is_pos2_poszero_negbminus1_reg0 = RegNext(is_pos2_poszero_negbminus1)
  val is_pos2_bminus1_b_reg0 = RegNext(is_pos2_bminus1_b)
  val is_pos2_b_bplus1_reg0 = RegNext(is_pos2_b_bplus1)
  val exp_normalized = Wire(UInt(5.W))
  val exp_normalized_reg0 = RegNext(exp_normalized)
  val sig_normalized = Wire(UInt(11.W))
  val sig_normalized_reg0 = RegNext(sig_normalized)
  val clz = Wire(UInt(4.W))
  val clz_reg0 = RegNext(clz)
  val out_exp_is_zero_negone = Wire(Bool())
  val out_exp_is_zero_negone_reg0 = RegNext(out_exp_is_zero_negone)
  val out_exp = Wire(UInt(5.W))
  val out_exp_reg0 = RegNext(out_exp)
  val out_sign = Wire(Bool())
  val out_sign_reg0 = RegNext(out_sign)

  /**
   * fp16 -> ui16  vfcvt.xu.f.v  vfcvt.rtz.xu.f.v
   *      -> i16   vfcvt.x.f.v   vfcvt.rtz.x.f.v
   *      -> ui8   vfncvt.xu.f.w vfncvt.rtz.xu.f.w
   *      -> i8    vfncvt.x.f.w  vfncvt.rtz.x.f.w
   */
  when(is_fp2int) {
    in_orR := 0.U
    exp_raw := 0.U
    sign := 0.U
    is_normal := 0.U
    is_inf := 0.U
    is_nan := 0.U
    is_neginf := 0.U
    is_neginf_negzero := 0.U
    is_negzero := 0.U
    is_poszero := 0.U
    is_snan := 0.U
    is_neg2_bplus1_b := 0.U
    is_neg2_b_bminus1 := 0.U
    is_neg2_negbminus1_negzero := 0.U
    is_pos2_poszero_negbminus1 := 0.U
    is_pos2_bminus1_b := 0.U
    is_pos2_b_bplus1 := 0.U
    exp_normalized := 0.U
    sig_normalized := 0.U
    clz := 0.U
    out_exp_is_zero_negone := 0.U
    out_exp := 0.U
    out_sign := 0.U

    val in = VectorFloat.fromUInt(io.src, f16.expWidth, f16.precision)
    raw_in := RawVectorFloat.fromVFP(in, Some(in.decode.expNotZero)).asUInt
    
    // single max exp = bias +& 15.U, narrow max exp = bias +& 7.U
    val max_int_exp = Mux(is_single, 30.U, 22.U)
    exp_of := raw_in.tail(1).head(5) > max_int_exp

    // left
    // only f16->i16(ui16) can left shift, shamt is raw_in.exp - (bias + precision - 1)
    val lpath_shamt = Mux(is_single, raw_in.tail(1).head(5) - 25.U, 0.U)
    // max shamt width = (15 - (precision - 1)).U.getWidth
    lpath_sig_shifted := Mux(is_single, (raw_in.tail(6) << lpath_shamt(4, 0))(15, 0), 0.U)
    // fp->ui16, if fp is negative, invalid
    lpath_iv := is_single && !is_signed_int && raw_in.head(1).asBool
    // f16->i16, may overflow
    val lpath_may_of = is_single && is_signed_int && (raw_in.tail(1).head(5) === max_int_exp)
    val lpath_pos_of = is_single && lpath_may_of && !raw_in.head(1).asBool
    val lpath_neg_of = is_single && lpath_may_of && raw_in.head(1).asBool && raw_in.tail(7).orR
    lpath_of := lpath_pos_of || lpath_neg_of

    // right
    // f16->i8(ui8) always right shift
    val rpath_shamt = Mux(is_single, 25.U - raw_in.tail(1).head(5), 22.U - raw_in.tail(1).head(5))
    val (rpath_sig_shifted, rpath_sticky) = ShiftRightJam(Cat(raw_in.tail(6), 0.U), rpath_shamt)
    rpath_sig_shifted0 := rpath_sig_shifted

    round_in := Mux(is_single, rpath_sig_shifted.head(f16.precision), rpath_sig_shifted.head(8))
    round_roundIn := Mux(is_single, rpath_sig_shifted.tail(f16.precision), rpath_sig_shifted.tail(8).head(1))
    round_stickyIn := rpath_sticky | Mux(is_narrow, rpath_sig_shifted.tail(9).orR, false.B)
    round_signIn := raw_in.head(1).asBool

    // select result
    sel_lpath := raw_in.tail(1).head(5) >= 25.U

    val max_int = Mux(is_single, Cat(!is_signed_int, ~0.U(15.W)), Cat(!is_signed_int, ~0.U(7.W)))
    val min_int = Mux(is_single, Cat(is_signed_int,   0.U(15.W)), Cat(is_signed_int,   0.U(7.W)))

    max_min_int := Mux(in.decode.isNaN | !raw_in.head(1).asBool, max_int, min_int)

  }.elsewhen(is_int2fp) {
    /**
     * ui16 -> f16  vfcvt.f.xu.v
     * i16  ->      vfcvt.f.x.v
     * ui8  ->      vfwcvt.f.xu.v
     * i8   ->      vfwcvt.f.x.v
     */
    raw_in := 0.U
    rpath_sig_shifted0 := 0.U
    sel_lpath := 0.U
    exp_of := 0.U
    lpath_of := 0.U
    lpath_iv := 0.U
    lpath_sig_shifted := 0.U
    max_min_int := 0.U
    is_normal := 0.U
    is_inf := 0.U
    is_nan := 0.U
    is_neginf := 0.U
    is_neginf_negzero := 0.U
    is_negzero := 0.U
    is_poszero := 0.U
    is_snan := 0.U
    is_neg2_bplus1_b := 0.U
    is_neg2_b_bminus1 := 0.U
    is_neg2_negbminus1_negzero := 0.U
    is_pos2_poszero_negbminus1 := 0.U
    is_pos2_bminus1_b := 0.U
    is_pos2_b_bplus1 := 0.U
    exp_normalized := 0.U
    sig_normalized := 0.U
    clz := 0.U
    out_exp_is_zero_negone := 0.U
    out_exp := 0.U
    out_sign := 0.U

    sign := is_signed_int && Mux(is_widen, io.src(7), io.src(15))

    val in_sext = Cat(Fill(8, io.src(7)), io.src(7,0))
    val in = Mux(is_signed_int && is_widen, in_sext, io.src)
    in_orR := in.orR
    val in_abs = Mux(sign, (~in).asUInt + 1.U, in)

    val lzc = CLZ(in_abs)
    val in_shift = (in_abs << lzc)(14, 0)
    exp_raw := 30.U - lzc

    round_in := Mux(is_widen, in_shift.head(8), in_shift.head(10))
    round_roundIn := Mux(is_widen, in_shift.tail(8).head(1), in_shift.tail(10).head(1)).asBool
    round_stickyIn := Mux(is_widen, in_shift.tail(9).orR, in_shift.tail(f16.precision).orR)
    round_signIn := sign

  }.elsewhen(is_vfr) {
    raw_in := 0.U
    rpath_sig_shifted0 := 0.U
    sel_lpath := 0.U
    exp_of := 0.U
    lpath_of := 0.U
    lpath_iv := 0.U
    lpath_sig_shifted := 0.U
    max_min_int := 0.U
    in_orR := 0.U
    exp_raw := 0.U
    round_in := 0.U
    round_roundIn := false.B
    round_stickyIn := false.B
    round_signIn := false.B

    val in = io.src
    sign := in.head(1).asBool
    val exp = in.tail(1).head(f16.expWidth)
    val sig = in.tail(6)

    is_normal := exp.orR & !exp.andR
    val is_subnormal = !exp.orR
    is_inf := exp.andR & !sig.orR
    is_nan := exp.andR & sig.orR
    is_neginf := sign & is_inf
    is_neginf_negzero := sign & (is_normal | is_subnormal & sig.orR)
    is_negzero := sign & is_subnormal & !sig.orR
    is_poszero := !sign & is_subnormal & !sig.orR
    val is_poszero_posinf = !sign & (is_normal | is_subnormal & sig.orR)
    is_snan := !sig.head(1).asBool & is_nan
    is_neg2_bplus1_b := sign & (exp === 30.U)
    is_neg2_b_bminus1 := sign & (exp === 29.U)
    is_neg2_negbminus1_negzero := sign & (sig.head(2) === "b00".U) & is_subnormal & sig.orR
    is_pos2_poszero_negbminus1 := !sign & (sig.head(2) === "b00".U) & is_subnormal & sig.orR
    is_pos2_bminus1_b := !sign & (exp === 29.U)
    is_pos2_b_bplus1 := !sign & (exp === 30.U)

     val zero_minus_lzc = 0.U - CLZ(sig) // 0 - count leading zero
     exp_normalized :=
       Mux(is_vfrsqrt7,
         Mux(is_poszero_posinf, Mux(is_normal, exp, Cat(Fill(f16.expWidth - zero_minus_lzc.getWidth, zero_minus_lzc.head(1)), zero_minus_lzc)), 0.U),
         Mux(is_normal, exp, Cat(Fill(f16.expWidth - zero_minus_lzc.getWidth, zero_minus_lzc.head(1)), zero_minus_lzc)))

     sig_normalized := Mux(is_vfrsqrt7, Mux(is_poszero_posinf, Mux(is_normal, Cat(0.U, sig), (sig << 1.U).asUInt), 0.U), Mux(is_normal, Cat(0.U, sig), (sig << 1.U).asUInt))

    clz := CLZ(sig_normalized)

    val out_exp_normalized = Mux(is_vfrec7, 29.U - exp_normalized, 0.U) // 2 * bias - 1 - exp_nor
    out_exp_is_zero_negone := !out_exp_normalized.orR || out_exp_normalized.andR

    out_exp := Mux(is_vfrsqrt7,
      Mux(is_normal, (44.U - exp) >> 1, (44.U + CLZ(sig)) >> 1), // if normal (3 * bias - 1 - exp) >> 1 else (3 * bias -1 + CLZ) >>1
      Mux(out_exp_is_zero_negone, 0.U, out_exp_normalized))

    out_sign := is_poszero_posinf & sign

  }.otherwise {
    raw_in := 0.U
    rpath_sig_shifted0 := 0.U
    sel_lpath := 0.U
    exp_of := 0.U
    lpath_of := 0.U
    lpath_iv := 0.U
    lpath_sig_shifted := 0.U
    max_min_int := 0.U
    in_orR := 0.U
    exp_raw := 0.U
    sign := 0.U
    round_in := 0.U
    round_roundIn := false.B
    round_stickyIn := false.B
    round_signIn := false.B
    is_normal := 0.U
    is_inf := 0.U
    is_nan := 0.U
    is_neginf := 0.U
    is_neginf_negzero := 0.U
    is_negzero := 0.U
    is_poszero := 0.U
    is_snan := 0.U
    is_neg2_bplus1_b := 0.U
    is_neg2_b_bminus1 := 0.U
    is_neg2_negbminus1_negzero := 0.U
    is_pos2_poszero_negbminus1 := 0.U
    is_pos2_bminus1_b := 0.U
    is_pos2_b_bplus1 := 0.U
    exp_normalized := 0.U
    sig_normalized := 0.U
    clz := 0.U
    out_exp_is_zero_negone := 0.U
    out_exp := 0.U
    out_sign := 0.U
  }

  when(is_fp2int_reg0) {
    val rpath_rounder = Module(new RoundingUnit(f16.precision))
    rpath_rounder.io.in := round_in_reg0
    rpath_rounder.io.roundIn := round_roundIn_reg0
    rpath_rounder.io.stickyIn := round_stickyIn_reg0
    rpath_rounder.io.signIn := round_signIn_reg0
    rpath_rounder.io.rm := rm_reg0

    val rpath_out_reg0 = Mux(rpath_rounder.io.r_up, rpath_rounder.io.in + 1.U, rpath_rounder.io.in)
    val rpath_cout_reg0 = rpath_rounder.io.r_up && Mux(is_narrow_reg0, rpath_rounder.io.in.tail(4).andR, rpath_rounder.io.in.andR)

    val rpath_sig_reg0 = Mux(is_single_reg0, Cat(0.U(4.W), rpath_cout_reg0, rpath_out_reg0), Cat(0.U(5.W), rpath_out_reg0))
    val rpath_ix_reg0 = rpath_rounder.io.inexact || (is_narrow_reg0 && rpath_sig_shifted_reg0.tail(8).orR)
    val rpath_iv_reg0 = !is_signed_int_reg0 && raw_in_reg0.head(1).asBool && rpath_sig_reg0.orR
    val rpath_pos_of_reg0 = !raw_in_reg0.head(1).asBool &&
      Mux(is_signed_int_reg0,
        (raw_in_reg0.tail(1).head(5) === 22.U) || ((raw_in_reg0.tail(1).head(5) === 21.U) && rpath_cout_reg0),
        (raw_in_reg0.tail(1).head(5) === 22.U) && rpath_cout_reg0)
    val rpath_neg_of_reg0 = raw_in_reg0.head(1).asBool && (raw_in_reg0.tail(1).head(5) === 22.U) && (rpath_rounder.io.in.tail(4).orR || rpath_rounder.io.r_up)
    val rpath_of_reg0 = Mux(is_narrow_reg0, rpath_pos_of_reg0 || rpath_neg_of_reg0, rpath_cout_reg0)

    val of = exp_of_reg0 || sel_lpath_reg0 && lpath_of_reg0 || !sel_lpath_reg0 && rpath_of_reg0
    val iv = of || sel_lpath_reg0 && lpath_iv_reg0 || !sel_lpath_reg0 && rpath_iv_reg0
    val ix = !iv && !sel_lpath_reg0 && rpath_ix_reg0

    val int_abs = Mux(sel_lpath_reg0, lpath_sig_shifted_reg0, rpath_sig_reg0)
    val int = Mux(is_narrow_reg0,
      Mux(raw_in_reg0.head(1).asBool && is_signed_int_reg0, -int_abs.tail(8), int_abs.tail(8)),
      Mux(raw_in_reg0.head(1).asBool && is_signed_int_reg0, -int_abs, int_abs))

    result0 := Mux(iv, max_min_int_reg0, int)
    fflags0 := Cat(iv, false.B, false.B, false.B, ix)
  }.elsewhen(is_int2fp_reg0) {
    val rounder = Module(new RoundingUnit(10))
    rounder.io.in := round_in_reg0
    rounder.io.roundIn := round_roundIn_reg0
    rounder.io.stickyIn := round_stickyIn_reg0
    rounder.io.signIn := round_signIn_reg0
    rounder.io.rm := rm_reg0

    val out_reg0 = Mux(rounder.io.r_up, rounder.io.in + 1.U, rounder.io.in)
    val cout_reg0 = rounder.io.r_up && rounder.io.in.andR.asBool

    val exp_reg0 = Mux(in_orR_reg0, exp_raw_reg0 + cout_reg0, 0.U)
    val sig_reg0 = out_reg0

    val of = exp_reg0 === 31.U
    val ix = rounder.io.inexact

    result0 := Cat(is_signed_int_reg0 && sign_reg0, exp_reg0, Mux(is_widen_reg0, Cat(sig_reg0.tail(2), 0.U(2.W)), sig_reg0))
    fflags0 := Cat(false.B, false.B, of, false.B, ix)
  }.elsewhen(is_vfr_reg0) {
    val vfrsqrt7Table = Module(new Rsqrt7Table)
    val vfrec7Table = Module(new Rec7Table)

    val sig_in7 = Mux(is_vfrsqrt7_reg0,
      Cat(exp_normalized_reg0(0), (sig_normalized_reg0 << Mux(is_normal_reg0, 0.U, clz_reg0))(9, 4)), // vfrsqrt7  Cat(exp_nor(0), sig_nor(9,4))
      (sig_normalized_reg0 << Mux(is_normal_reg0, 0.U, clz_reg0))(9, 3)) // vfrec7 sig_nor(9,3)

    vfrsqrt7Table.src := sig_in7
    vfrec7Table.src := sig_in7

    val sig_out7 = Wire(UInt(7.W))
    sig_out7 := Mux(is_vfrsqrt7_reg0, vfrsqrt7Table.out, vfrec7Table.out)

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
  }.otherwise {
    result0 := 0.U
  }


  result := result0_reg0
  fflags := fflags0_reg0

  io.result := result
  io.fflags := fflags
  */


  /** cycle0                                                              | cycle1                  |   cycle2
   * fp2int   in(16) raw_in(17)  left,right    ShiftRightJam(37)          | RoundingUnit(11)  adder |
   * int2fp   in(16) in_abs(16)  lzc in_shift  exp_raw                    | RoundingUnit(10)  adder |  -> result & fflags
   * vfr      in(16)             lzc exp_nor   sig_nor  clz out_exp adder | Table                   |
   *
   */
  // control path
  val is_sew_8 = io.sew === "b00".U
  val is_sew_16 = io.sew === "b01".U
  val is_single = io.opType.tail(3).head(2) === "b00".U
  val is_widen = io.opType.tail(3).head(2) === "b01".U
  val is_narrow = io.opType.tail(3).head(2) === "b10".U
  val is_single_reg0 = RegNext(is_single, false.B)
  val is_widen_reg0 = RegNext(is_widen, false.B)
  val is_narrow_reg0 = RegNext(is_narrow, false.B)

  val is_vfr = io.opType(5).asBool && is_sew_16
  val is_fp2int = io.opType.head(2) === "b10".U && (is_sew_16 && is_single || is_sew_8 && is_narrow)
  val is_int2fp = io.opType.head(2) === "b01".U && (is_sew_8 && is_widen || is_sew_16 && is_single)
  val is_vfr_reg0 = RegNext(is_vfr, false.B)
  val is_fp2int_reg0 = RegNext(is_fp2int, false.B)
  val is_int2fp_reg0 = RegNext(is_int2fp, false.B)

  val is_vfrsqrt7 = is_vfr && !io.opType(0).asBool
  val is_vfrec7 = is_vfr && io.opType(0).asBool
  val is_vfrsqrt7_reg0 = RegNext(is_vfrsqrt7, false.B)
  val is_vfrec7_reg0 = RegNext(is_vfrec7, false.B)

  val is_signed_int = io.opType(0)
  val is_signed_int_reg0 = RegNext(is_signed_int, false.B)

  val result = Wire(UInt(16.W))
  val NV, DZ, OF, UF, NX = WireInit(false.B)
  val fflags = WireInit(Cat(NV, DZ, OF, UF, NX))

  val result0 = Wire(UInt(16.W))
  val result0_reg0 = RegNext(result0, 0.U(16.W))
  val fflags0 = WireInit(Cat(NV, DZ, OF, UF, NX))
  val fflags0_reg0 = RegNext(fflags0)

  val round_in = Wire(UInt(11.W))
  val round_roundIn = Wire(Bool())
  val round_stickyIn = Wire(Bool())
  val round_signIn = Wire(Bool())
  val round_in_reg0 = RegNext(round_in, 0.U(11.W))
  val round_roundIn_reg0 = RegNext(round_roundIn, false.B)
  val round_stickyIn_reg0 = RegNext(round_stickyIn, false.B)
  val round_signIn_reg0 = RegNext(round_signIn, false.B)
  val rm_reg0 = RegNext(io.rm)

  val is_normal = Wire(Bool())
  val is_inf = Wire(Bool())
  val is_nan = Wire(Bool())
  val is_neginf = Wire(Bool())
  val is_neginf_negzero = Wire(Bool())
  val is_negzero = Wire(Bool())
  val is_poszero = Wire(Bool())
  val is_snan = Wire(Bool())
  val is_neg2_bplus1_b = Wire(Bool())
  val is_neg2_b_bminus1 = Wire(Bool())
  val is_neg2_negbminus1_negzero = Wire(Bool())
  val is_pos2_poszero_negbminus1 = Wire(Bool())
  val is_pos2_bminus1_b = Wire(Bool())
  val is_pos2_b_bplus1 = Wire(Bool())

  val is_normal_reg0 = RegNext(is_normal, false.B)
  val is_inf_reg0 = RegNext(is_inf, false.B)
  val is_nan_reg0 = RegNext(is_nan, false.B)
  val is_neginf_reg0 = RegNext(is_neginf, false.B)
  val is_neginf_negzero_reg0 = RegNext(is_neginf_negzero, false.B)
  val is_negzero_reg0 = RegNext(is_negzero, false.B)
  val is_poszero_reg0 = RegNext(is_poszero, false.B)
  val is_snan_reg0 = RegNext(is_snan, false.B)
  val is_neg2_bplus1_b_reg0 = RegNext(is_neg2_bplus1_b, false.B)
  val is_neg2_b_bminus1_reg0 = RegNext(is_neg2_b_bminus1, false.B)
  val is_neg2_negbminus1_negzero_reg0 = RegNext(is_neg2_negbminus1_negzero, false.B)
  val is_pos2_poszero_negbminus1_reg0 = RegNext(is_pos2_poszero_negbminus1, false.B)
  val is_pos2_bminus1_b_reg0 = RegNext(is_pos2_bminus1_b, false.B)
  val is_pos2_b_bplus1_reg0 = RegNext(is_pos2_b_bplus1, false.B)

  // cycle 0
  // in
  val in_sext = Wire(UInt(17.W))
  in_sext := Mux1H(
    Seq(is_int2fp && !is_signed_int && is_widen,
      is_int2fp && is_signed_int && is_widen,
      is_int2fp && !is_signed_int && is_single,
      is_int2fp && is_signed_int && is_single),
    Seq(Cat(Fill(9, 0.U), io.src(7,0)),
      Cat(Fill(9, io.src(7)), io.src(7,0)),
      Cat(0.U, io.src),
      Cat(io.src(15), io.src))
  )
  val in_orR = Wire(Bool())
  val in_orR_reg0 = RegNext(in_orR, false.B)
  in_orR := in_sext.orR

  val fp_in = VectorFloat.fromUInt(io.src, f16.expWidth, f16.precision)
  val fp2int_in = Wire(UInt(17.W))
  fp2int_in := RawVectorFloat.fromVFP(fp_in, Some(fp_in.decode.expNotZero)).asUInt

  val in = Wire(UInt(17.W))
  val in_reg0 = RegNext(in, 0.U(17.W))
  in := Mux1H(
    Seq(is_fp2int,
      is_int2fp,
      is_vfr),
    Seq(fp2int_in,
      in_sext,
      Cat(0.U, io.src))
  )

  val sign = Wire(Bool())
  val sign_reg0 = RegNext(sign, false.B)
  sign := Mux1H(
    Seq(is_fp2int || is_int2fp,
      is_vfr),
    Seq(in.head(1).asBool,
      in.tail(1).head(1).asBool)
  )


  val exp = Mux1H(
    Seq(is_fp2int,
      is_vfr),
    Seq(in.tail(1).head(f16.expWidth),
      in.tail(2).head(f16.expWidth))
  )
  val sig = Mux1H(
    Seq(is_fp2int,
      is_vfr),
    Seq(in.tail(6),
      in.tail(7))
  )

  // fp2int
  // left
  val max_int_exp = Mux1H(
    Seq(is_fp2int && is_single,
      is_fp2int && is_narrow),
    Seq(30.U,
      22.U)
  )

  val exp_of = Wire(Bool())
  val exp_of_reg0 = RegNext(exp_of, false.B)
  exp_of := is_fp2int && (exp > max_int_exp)
  
  val lpath_shamt = Mux(is_fp2int && is_single, exp - 25.U, 0.U)
  val lpath_sig_shifted = Wire(UInt(16.W))
  val lpath_sig_shifted_reg0 = RegNext(lpath_sig_shifted, 0.U(16.W))
  lpath_sig_shifted := Mux(is_fp2int && is_single, (sig << lpath_shamt(4, 0))(15, 0), 0.U)

  val lpath_iv = Wire(Bool())
  val lpath_iv_reg0 = RegNext(lpath_iv, false.B)
  lpath_iv := is_fp2int && is_single && !is_signed_int && sign


  val lpath_of = Wire(Bool())
  val lpath_of_reg0 = RegNext(lpath_of)
  lpath_of := is_fp2int && is_single && is_signed_int && (exp === max_int_exp) && (!sign || (sign && in.tail(7).orR))


  // right
  val rpath_shamt = Mux1H(
    Seq(is_fp2int && is_single,
      is_fp2int && is_narrow),
    Seq(25.U - exp,
    22.U - exp)
  )

  val rpath_sig_shifted0 = Wire(UInt(12.W))
  val rpath_sig_shifted_reg0 = RegNext(rpath_sig_shifted0, 0.U(12.W))
  val (rpath_sig_shifted, rpath_sticky) = ShiftRightJam(Cat(sig, 0.U), rpath_shamt)
  rpath_sig_shifted0 := rpath_sig_shifted


  // int2fp
  val in_abs = Mux1H(
    Seq(is_int2fp && sign,
      is_int2fp && !sign),
    Seq((~in).asUInt + 1.U,
      in)
  )

  val int2fp_clz = Mux(is_int2fp, CLZ(in_abs(15,0)), 0.U)

  val in_shift = Mux(is_int2fp, (in_abs << int2fp_clz)(14, 0), 0.U)

  val exp_raw = Wire(UInt(5.W))
  val exp_raw_reg0 = RegNext(exp_raw, 0.U(5.W))
  exp_raw := Mux(is_int2fp, 30.U - int2fp_clz, 0.U)



  // share RoundingUnit
  round_in := Mux1H(
    Seq(is_fp2int && is_single,
      is_fp2int && is_narrow,
      is_int2fp && is_widen,
      is_int2fp && is_single),
    Seq(rpath_sig_shifted.head(f16.precision),
      rpath_sig_shifted.head(8),
      in_shift.head(8),
      in_shift.head(10))
  )
  round_roundIn := Mux1H(
    Seq(is_fp2int && is_single,
      is_fp2int && is_narrow,
      is_int2fp && is_widen,
      is_int2fp && is_single),
    Seq(rpath_sig_shifted.tail(f16.precision),
      rpath_sig_shifted.tail(8).head(1),
      in_shift.tail(8).head(1),
      in_shift.tail(10).head(1))
  )
  round_stickyIn := Mux1H(
    Seq(is_fp2int && is_single,
      is_fp2int && is_narrow,
      is_int2fp && is_widen,
      is_int2fp && is_single),
    Seq(rpath_sticky,
      rpath_sticky || rpath_sig_shifted.tail(9).orR,
      in_shift.tail(9).orR,
      in_shift.tail(f16.precision).orR)
  )
  round_signIn := sign

  val sel_lpath = Wire(Bool())
  val sel_lpath_reg0 = RegNext(sel_lpath)
  sel_lpath := exp >= 25.U


  val max_int = Mux1H(
    Seq(is_fp2int && is_single,
      is_fp2int && is_narrow),
    Seq(Cat(!is_signed_int, ~0.U(15.W)),
      Cat(!is_signed_int,   ~0.U(7.W)))
  )
  val min_int = Mux1H(
    Seq(is_fp2int && is_single,
      is_fp2int && is_narrow),
    Seq(Cat(is_signed_int, 0.U(15.W)),
      Cat(is_signed_int,   0.U(7.W)))
  )
  val max_min_int = Wire(UInt(16.W))
  val max_min_int_reg0 = RegNext(max_min_int, 0.U(16.W))
  max_min_int := Mux(exp.andR && sig.tail(1).orR || !sign, max_int, min_int)


  // vfr
  is_normal := exp.orR & !exp.andR
  val is_subnormal = !exp.orR
  is_inf := exp.andR & !sig.tail(1).orR
  is_nan := exp.andR & sig.tail(1).orR
  is_neginf := sign & is_inf
  is_neginf_negzero := sign & (is_normal | is_subnormal & sig.tail(1).orR)
  is_negzero := sign & is_subnormal & !sig.tail(1).orR
  is_poszero := !sign & is_subnormal & !sig.tail(1).orR
  val is_poszero_posinf = !sign & (is_normal | is_subnormal & sig.tail(1).orR)
  is_snan := !sig.tail(1).head(1).asBool & is_nan
  is_neg2_bplus1_b := sign & (exp === 30.U)
  is_neg2_b_bminus1 := sign & (exp === 29.U)
  is_neg2_negbminus1_negzero := sign & (sig.tail(1).head(2) === "b00".U) & is_subnormal & sig.tail(1).orR
  is_pos2_poszero_negbminus1 := !sign & (sig.tail(1).head(2) === "b00".U) & is_subnormal & sig.tail(1).orR
  is_pos2_bminus1_b := !sign & (exp === 29.U)
  is_pos2_b_bplus1 := !sign & (exp === 30.U)

  val zero_minus_lzc = Mux(is_vfr, 0.U - CLZ(sig.tail(1)), 0.U)

  val exp_normalized = Wire(UInt(5.W))
  val exp_normalized_reg0 = RegNext(exp_normalized)
  exp_normalized := Mux1H(
    Seq(is_vfrsqrt7 && is_poszero_posinf && is_normal || (is_vfrec7 && is_normal),
      is_vfrsqrt7 && is_poszero_posinf && is_subnormal || (is_vfrec7 && is_subnormal)),
    Seq(exp,
      Cat(Fill(f16.expWidth - zero_minus_lzc.getWidth, zero_minus_lzc.head(1)), zero_minus_lzc))
  )


  val sig_normalized = Wire(UInt(11.W))
  val sig_normalized_reg0 = RegNext(sig_normalized)
  sig_normalized := Mux1H(
    Seq(is_vfrsqrt7 && is_poszero_posinf && is_normal || (is_vfrec7 && is_normal),
      is_vfrsqrt7 && is_poszero_posinf && is_subnormal || (is_vfrec7 && is_subnormal)),
    Seq(Cat(0.U, sig.tail(1)),
      (sig.tail(1) << 1.U).asUInt)
  )


  val clz_sig = Wire(UInt(4.W))
  val clz_sig_reg0 = RegNext(clz_sig)
  clz_sig := CLZ(sig_normalized)


  val out_exp_normalized = Mux(is_vfrec7, 29.U - exp_normalized, 0.U)

  val out_exp_zero_negone = Wire(Bool())
  val out_exp_zero_negone_reg0 = RegNext(out_exp_zero_negone)
  out_exp_zero_negone := is_vfrec7 && !out_exp_normalized.orR || out_exp_normalized.andR


  val out_exp = Wire(UInt(5.W))
  val out_exp_reg0 = RegNext(out_exp, 0.U(5.W))
  out_exp := Mux1H(
    Seq(is_vfrsqrt7 && is_normal,
      is_vfrsqrt7 && is_subnormal,
      is_vfrec7 && out_exp_zero_negone,
      is_vfrec7 && !out_exp_zero_negone
    ),
    Seq((44.U - exp) >> 1,
      (44.U + CLZ(sig.tail(1))) >> 1,
      0.U,
      out_exp_normalized)
  )


  val out_sign = Wire(Bool())
  val out_sign_reg0 = RegNext(out_sign)
  out_sign := is_vfrsqrt7 && is_poszero_posinf && sign


  // cycle1
  val rounder = Module(new RoundingUnit(f16.precision))
  rounder.io.in := round_in_reg0
  rounder.io.roundIn := round_roundIn_reg0
  rounder.io.stickyIn := round_stickyIn_reg0
  rounder.io.signIn := round_signIn_reg0
  rounder.io.rm := rm_reg0

  val out_reg0 = Mux(rounder.io.r_up, rounder.io.in + 1.U, rounder.io.in)
  val cout_reg0 = rounder.io.r_up && Mux1H(
    Seq(is_fp2int_reg0 && is_narrow_reg0,
      is_fp2int_reg0 && is_single_reg0,
      is_int2fp_reg0),
    Seq(rounder.io.in.tail(4).andR,
      rounder.io.in.andR,
      rounder.io.in.tail(1).andR)
  )
  val exp_reg0 = Mux1H(
    Seq(is_int2fp_reg0 && in_orR_reg0,
      is_int2fp_reg0 && !in_orR_reg0),
    Seq(exp_raw_reg0 + cout_reg0,
      0.U)
  )
  val sig_reg0 = Mux1H(
    Seq(is_fp2int_reg0 && is_single_reg0,
      is_fp2int_reg0 && is_narrow_reg0,
      is_int2fp_reg0),
    Seq(Cat(0.U(4.W), cout_reg0, out_reg0),
      Cat(0.U(5.W), out_reg0),
      out_reg0)
  )
  val rpath_ix_reg0 = Mux1H(
    Seq(is_fp2int_reg0 && is_single_reg0,
      is_fp2int_reg0 && is_narrow_reg0),
    Seq(rounder.io.inexact,
      rounder.io.inexact || rpath_sig_shifted_reg0.tail(8).orR)
  )
  val rpath_iv_reg0 = is_fp2int_reg0 && !is_signed_int_reg0 && in_reg0.head(1).asBool && sig_reg0.orR

  val rpath_pos_of_reg0 = Mux1H(
    Seq(is_fp2int_reg0 && !in_reg0.head(1).asBool && is_signed_int_reg0,
      is_fp2int_reg0 && !in_reg0.head(1).asBool && !is_signed_int_reg0),
    Seq((in_reg0.tail(1).head(5) === 22.U) || ((in_reg0.tail(1).head(5) === 21.U) && cout_reg0),
      (in_reg0.tail(1).head(5) === 22.U) && cout_reg0)
  )
  val rpath_neg_of_reg0 = is_fp2int_reg0 && in_reg0.head(1).asBool && (in_reg0.tail(1).head(5) === 22.U) && (rounder.io.in.tail(4).orR || rounder.io.r_up)

  val rpath_of_reg0 = Mux1H(
    Seq(is_fp2int_reg0 && is_narrow_reg0,
      is_fp2int_reg0 && is_single_reg0),
    Seq(rpath_pos_of_reg0 || rpath_neg_of_reg0,
      cout_reg0)
  )


  val vfrsqrt7Table = Module(new Rsqrt7Table)
  val vfrec7Table = Module(new Rec7Table)

  val sig_in7 = Mux1H(
    Seq(is_vfr_reg0 && is_vfrsqrt7_reg0,
      is_vfr_reg0 && is_vfrec7_reg0),
    Seq(Cat(exp_normalized_reg0(0), (sig_normalized_reg0 << Mux(is_normal_reg0, 0.U, clz_sig_reg0))(9, 4)),
      (sig_normalized_reg0 << Mux(is_normal_reg0, 0.U, clz_sig_reg0))(9, 3))
  )
  vfrsqrt7Table.src := sig_in7
  vfrec7Table.src := sig_in7

  val sig_out7_reg0 = Wire(UInt(7.W))
  sig_out7_reg0 := Mux1H(
    Seq(is_vfr_reg0 && is_vfrsqrt7_reg0,
      is_vfr_reg0 && is_vfrec7_reg0),
    Seq(vfrsqrt7Table.out,
      vfrec7Table.out)
  )

  val out_sig_reg0 = Mux1H(
    Seq(is_vfrec7_reg0 && out_exp_zero_negone_reg0 && (is_neg2_bplus1_b_reg0 || is_pos2_b_bplus1_reg0),
      is_vfrec7_reg0 && out_exp_zero_negone_reg0 && (is_neg2_b_bminus1_reg0 || is_pos2_bminus1_b_reg0),
      is_vfrec7_reg0 && out_exp_zero_negone_reg0 && !(is_neg2_bplus1_b_reg0 || is_pos2_b_bplus1_reg0 || is_neg2_b_bminus1_reg0 || is_pos2_bminus1_b_reg0),
      is_vfrec7_reg0 && !out_exp_zero_negone_reg0),
    Seq(Cat(0.U, 1.U, sig_out7_reg0, 0.U),
      Cat(1.U, sig_out7_reg0, 0.U(2.W)),
      Cat(1.U, sig_out7_reg0, 0.U(2.W)) >> 1,
      Cat(sig_out7_reg0, 0.U(3.W)))
  )

  val fp_result = Wire(UInt(16.W))
  fp_result := Mux1H(
    Seq(is_vfrsqrt7_reg0,
      is_vfrec7_reg0),
    Seq(Cat(out_sign_reg0, out_exp_reg0, sig_out7_reg0, 0.U(3.W)),
      Cat(sign_reg0, out_exp_reg0, out_sig_reg0))
  )

  val result_nan = Cat(0.U(1.W), Fill(6, 1.U), 0.U(9.W))
  val result_inf = Cat(Fill(5, 1.U), 0.U(10.W))
  val result_greatest_fin = Cat(Fill(4, 1.U), 0.U, Fill(10, 1.U))


  val of_reg0 = Mux1H(
    Seq(is_fp2int_reg0,
      is_int2fp_reg0),
    Seq(exp_of_reg0 || sel_lpath_reg0 && lpath_of_reg0 || !sel_lpath_reg0 && rpath_of_reg0,
      exp_reg0 === 31.U)
  )
  val iv_reg0 = is_fp2int_reg0 && (of_reg0 || sel_lpath_reg0 && lpath_iv_reg0 || !sel_lpath_reg0 && rpath_iv_reg0)

  val ix_reg0 = Mux1H(
    Seq(is_fp2int_reg0,
      is_int2fp_reg0),
    Seq(!iv_reg0 && !sel_lpath_reg0 && rpath_ix_reg0,
      rounder.io.inexact)
  )

  val int_abs_reg0 = Mux1H(
    Seq(is_fp2int_reg0 && sel_lpath_reg0,
      is_fp2int_reg0 && !sel_lpath_reg0),
    Seq(lpath_sig_shifted_reg0,
      sig_reg0)
  )
  val int_reg0 = Mux1H(
    Seq(is_fp2int_reg0 && is_narrow_reg0,
      is_fp2int_reg0 && is_single_reg0),
    Seq(Mux(in_reg0.head(1).asBool && is_signed_int_reg0, -int_abs_reg0.tail(8), int_abs_reg0.tail(8)),
      Mux(in_reg0.head(1).asBool && is_signed_int_reg0, -int_abs_reg0, int_abs_reg0))
  )

  result0 := Mux1H(
    Seq(is_fp2int_reg0,
      is_int2fp_reg0,
      is_vfrsqrt7_reg0 && (is_nan_reg0 || is_neginf_negzero_reg0) || is_vfrec7_reg0 && is_nan_reg0,
      is_vfrsqrt7_reg0 && is_inf_reg0,
      is_vfrsqrt7_reg0 && (is_negzero_reg0 || is_poszero_reg0),
      is_vfrsqrt7_reg0 && !(is_nan_reg0 || is_neginf_negzero_reg0 || is_inf_reg0 || is_negzero_reg0 || is_poszero_reg0) ||
        is_vfrec7_reg0 && !(is_nan_reg0 || is_inf_reg0 || is_negzero_reg0 || is_poszero_reg0 || is_neg2_negbminus1_negzero_reg0 || is_pos2_poszero_negbminus1_reg0),
      is_vfrec7_reg0 && is_inf_reg0,
      is_vfrec7_reg0 && (is_negzero_reg0 || is_poszero_reg0),
      is_vfrec7_reg0 && is_neg2_negbminus1_negzero_reg0,
      is_vfrec7_reg0 && is_pos2_poszero_negbminus1_reg0
    ),
    Seq(Mux(iv_reg0, max_min_int_reg0, int_reg0),
      Cat(is_signed_int_reg0 && sign_reg0, exp_reg0, Mux(is_widen_reg0, Cat(sig_reg0(7,0), 0.U(2.W)), sig_reg0(9,0))),
      result_nan,
      Mux(is_neginf_reg0, result_nan, 0.U),
      Mux(is_negzero_reg0, Cat(1.U, result_inf), Cat(0.U, result_inf)),
      fp_result,
      Mux(is_neginf_reg0, Cat(1.U, 0.U(15.W)), 0.U),
      Mux(is_negzero_reg0, Cat(Fill(6, 1.U), 0.U(10.W)), Cat(0.U(1.W), Fill(5, 1.U), 0.U(10.W))),
      Mux(rm_reg0 === RUP || rm_reg0 === RTZ, Cat(1.U, result_greatest_fin), Cat(1.U, result_inf)),
      Mux(rm_reg0 === RDN || rm_reg0 === RTZ, Cat(0.U, result_greatest_fin), Cat(0.U, result_inf)))
  )

  fflags0 := Mux1H(
    Seq(is_fp2int_reg0,
      is_int2fp_reg0,
      is_vfrsqrt7_reg0 && (is_nan_reg0 || is_neginf_negzero_reg0),
      is_vfrsqrt7_reg0 && is_inf_reg0,
      is_vfrsqrt7_reg0 && (is_negzero_reg0 || is_poszero_reg0) || is_vfrec7_reg0 && (is_negzero_reg0 || is_poszero_reg0),
      is_vfrsqrt7_reg0 && !(is_nan_reg0 || is_neginf_negzero_reg0 || is_inf_reg0 || is_negzero_reg0 || is_poszero_reg0) ||
        is_vfrec7_reg0 && !(is_nan_reg0 || is_inf_reg0 || is_negzero_reg0 || is_poszero_reg0 || is_neg2_negbminus1_negzero_reg0 || is_pos2_poszero_negbminus1_reg0),
      is_vfrec7_reg0 && is_nan_reg0,
      is_vfrec7_reg0 && is_inf_reg0,
      is_vfrec7_reg0 && (is_neg2_negbminus1_negzero_reg0 || is_pos2_poszero_negbminus1_reg0)),
    Seq(Cat(iv_reg0, false.B, false.B, false.B, ix_reg0),
      Cat(false.B, false.B, of_reg0, false.B, ix_reg0),
      Mux(is_snan_reg0 || is_neginf_negzero_reg0, "b10000".U, "b00000".U),
      Mux(is_neginf_reg0, "b10000".U, "b00000".U),
      "b01000".U,
      0.U,
      Mux(is_snan_reg0, "b10000".U, "b00000".U),
      Mux(is_neginf_reg0, Cat(1.U, 0.U(15.W)), 0.U),
      "b00101".U)
  )

  // cycle2
  result := result0_reg0
  fflags := fflags0_reg0

  io.result := result
  io.fflags := fflags
}
