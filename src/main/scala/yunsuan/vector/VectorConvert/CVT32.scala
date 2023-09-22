package yunsuan.vector.VectorConvert

import chisel3._
import chisel3.util._
import yunsuan.util.LookupTree
import yunsuan.vector.VectorConvert.util.{CLZ, ShiftRightJam, VFRSqrtTable, VFRecTable, RoundingUnit}
import yunsuan.vector.VectorConvert.RoundingModle._


class CVT32(width: Int = 32) extends CVT(width){
  /*
  val is_sew_16 = io.sew === "b01".U
  val is_sew_32 = io.sew === "b10".U

  val is_single = io.opType.tail(3).head(2) === "b00".U
  val is_widen  = io.opType.tail(3).head(2) === "b01".U
  val is_narrow = io.opType.tail(3).head(2) === "b10".U
  val is_single_reg0 = RegNext(is_single)
  val is_widen_reg0 = RegNext(is_widen)
  val is_narrow_reg0 = RegNext(is_narrow)


  val is_vfr = io.opType(5).asBool
  val is_fp2int = io.opType.head(2) === "b10".U
  val is_int2fp = io.opType.head(2) === "b01".U
  val is_fp2fp = io.opType.head(3) === "b110".U


  val is_signed_int = io.opType(0)
  val is_signed_int_reg0 = RegNext(is_signed_int)

  val result = Wire(UInt(32.W))
  val NV, DZ, OF, UF, NX = WireInit(false.B)
  val fflags = WireInit(Cat(NV, DZ, OF, UF, NX))

  val result0 = Wire(UInt(32.W))
  val result0_reg0 = RegNext(result0, 0.U(32.W))
  val fflags0 = WireInit(Cat(NV, DZ, OF, UF, NX))
  val fflags0_reg0 = RegNext(fflags0)

  val round_in = Wire(UInt(24.W))
  val round_in_reg0 = RegNext(round_in)
  val round_roundIn = Wire(Bool())
  val round_roundIn_reg0 = RegNext(round_roundIn)
  val round_stickyIn = Wire(Bool())
  val round_stickyIn_reg0 = RegNext(round_stickyIn)
  val round_signIn = Wire(Bool())
  val round_signIn_reg0 = RegNext(round_signIn)
  val rm_reg0 = RegNext(io.rm)


  // fp2int
  val fp32toint32 = Wire(Bool())
  val fp32toint16 = Wire(Bool())
  val fp16toint32 = Wire(Bool())
  val fp16toint16 = Wire(Bool())
  val fp16toint8  = Wire(Bool())

  val fp32toint32_reg0 = RegNext(fp32toint32)
  val fp32toint16_reg0 = RegNext(fp32toint16)
  val fp16toint32_reg0 = RegNext(fp16toint32)
  val fp16toint16_reg0 = RegNext(fp16toint16)
  val fp16toint8_reg0 = RegNext(fp16toint8)

  val raw_in = Wire(UInt(33.W))
  val raw_in_reg0 = RegNext(raw_in, 0.U(33.W))
  val rpath_sig_shifted0 = Wire(UInt(64.W))
  val rpath_sig_shifted_reg0 = RegNext(rpath_sig_shifted0)
  val sel_lpath = Wire(Bool())
  val sel_lpath_reg0 = RegNext(sel_lpath)
  val exp_of = Wire(Bool())
  val exp_of_reg0 = RegNext(exp_of)
  val lpath_of = Wire(Bool())
  val lpath_of_reg0 = RegNext(lpath_of)
  val lpath_iv = Wire(Bool())
  val lpath_iv_reg0 = RegNext(lpath_iv)
  val lpath_sig_shifted = Wire(UInt(32.W))
  val lpath_sig_shifted_reg0 = RegNext(lpath_sig_shifted)
  val sign_or_nan = Wire(Bool())
  val sign_or_nan_reg0 = RegNext(sign_or_nan)

  // int2fp
  val out_is_fp32 = Wire(Bool())
  val int32tofp32 = Wire(Bool())
  val int16tofp32 = Wire(Bool())
  val int32tofp16 = Wire(Bool())
  val int16tofp16 = Wire(Bool())
  val int8tofp16  = Wire(Bool())
  val out_is_fp32_reg0 = RegNext(out_is_fp32)
  val int32tofp32_reg0 = RegNext(int32tofp32)
  val int16tofp32_reg0 = RegNext(int16tofp32)
  val int32tofp16_reg0 = RegNext(int32tofp16)
  val int16tofp16_reg0 = RegNext(int16tofp16)
  val int8tofp16_reg0 = RegNext(int8tofp16)
  val in_orR = Wire(Bool())
  val in_orR_reg0 = RegNext(in_orR)
  val exp_raw = Wire(UInt(8.W))
  val exp_raw_reg0 = RegNext(exp_raw)
  val sign = Wire(Bool())
  val sign_reg0 = RegNext(sign)

  // vfr
  val is_vfrsqrt7 = is_vfr && !io.opType(0).asBool
  val is_vfrec7 = is_vfr && io.opType(0).asBool
  val is_vfrsqrt7_reg0 = RegNext(is_vfrsqrt7)
  val is_vfrec7_reg0 = RegNext(is_vfrec7)
  val in_is_fp16 = Wire(Bool())
  val in_is_fp16_reg0 = RegNext(in_is_fp16)
  val in_is_fp32 = Wire(Bool())
  val in_is_fp32_reg0 = RegNext(in_is_fp32)

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
  val exp_normalized_5 = Wire(UInt(5.W))
  val exp_normalized0_reg0 = RegNext(exp_normalized_5)
  val exp_normalized_8 = Wire(UInt(8.W))
  val exp_normalized1_reg0 = RegNext(exp_normalized_8)
  val sig_normalized_11 = Wire(UInt(11.W))
  val sig_normalized0_reg0 = RegNext(sig_normalized_11)
  val sig_normalized_24 = Wire(UInt(24.W))
  val sig_normalized1_reg0 = RegNext(sig_normalized_24)
  val clz = Wire(UInt(8.W))
  val clz_reg0 = RegNext(clz)
  val out_exp_zero_negone = Wire(Bool())
  val out_exp_zero_negone_reg0 = RegNext(out_exp_zero_negone)
  val out_exp_16 = Wire(UInt(5.W))
  val out_exp_16_reg0 = RegNext(out_exp_16)
  val out_exp_32 = Wire(UInt(8.W))
  val out_exp_32_reg0 = RegNext(out_exp_32)
  val out_sign = Wire(Bool())
  val out_sign_reg0 = RegNext(out_sign)

  // fp2fp
  val down_exp = Wire(UInt(8.W))
  val down_exp_reg0 = RegNext(down_exp)
  val nor_sig = Wire(UInt(10.W))
  val nor_sig_reg0 = RegNext(nor_sig)
  val nor_roundBit = Wire(Bool())
  val nor_roundBit_reg0 = RegNext(nor_roundBit)
  val nor_stickyBit = Wire(Bool())
  val nor_stickyBit_reg0 = RegNext(nor_stickyBit)
  val subnor_in = Wire(UInt(10.W))
  val subnor_in_reg0 = RegNext(subnor_in)
  val subnor_roundBit = Wire(Bool())
  val subnor_roundBit_reg0 = RegNext(subnor_roundBit)
  val subnor_stickyBit = Wire(Bool())
  val subnor_stickyBit_reg0 = RegNext(subnor_stickyBit)
  val may_be_subnor = Wire(Bool())
  val may_be_subnor_reg0 = RegNext(may_be_subnor)
  val special_case = Wire(Bool())
  val special_case_reg0 = RegNext(special_case)
  val nor_of_exp = Wire(UInt(5.W))
  val nor_of_exp_reg0 = RegNext(nor_of_exp)
  val nor_of_sig = Wire(UInt(24.W))
  val nor_of_sig_reg0 = RegNext(nor_of_sig)
  val in_isnan_neg = Wire(Bool())
  val in_isnan_neg_reg0 = RegNext(in_isnan_neg)
  val in_sigNotZero = Wire(Bool())
  val in_sigNotZero_reg0 = RegNext(in_sigNotZero)
  val iv = Wire(Bool())
  val iv_reg0 = RegNext(iv)
  val dz = Wire(Bool())
  val dz_reg0 = RegNext(dz)
  val nor_exp = Wire(UInt(8.W))
  val nor_exp_reg0 = RegNext(nor_exp)
  val nor_sig_24 = Wire(UInt(10.W))
  val nor_sig_24_reg0 = RegNext(nor_sig_24)
  val subnor_exp = Wire(UInt(8.W))
  val subnor_exp_reg0 = RegNext(subnor_exp)
  val subnor_sig = Wire(UInt(24.W))
  val subnor_sig_reg0 = RegNext(subnor_sig)
  val isNaN = Wire(Bool())
  val isNaN_reg0 = RegNext(isNaN)
  val expIsOnes = Wire(Bool())
  val expIsOnes_reg0 = RegNext(expIsOnes)
  val expIsZero = Wire(Bool())
  val expIsZero_reg0 = RegNext(expIsZero)
  val isZero = Wire(Bool())
  val isZero_reg0 = RegNext(isZero)
  val isSubnormal = Wire(Bool())
  val isSubnormal_reg0 = RegNext(isSubnormal)
  val sigNotZero = Wire(Bool())
  val sigNotZero_reg0 = RegNext(sigNotZero)
  /**
   * fptoint
   * fp32 -> ui32
   * ->  i32
   * -> ui16
   * ->  i16
   *
   * fp16 -> ui16
   * ->  i16
   * -> ui32
   * ->  i32
   * -> ui8
   * ->  i8
   */
  when(is_fp2int) {
    out_is_fp32 := 0.U
    int32tofp32 := 0.U
    int16tofp32 := 0.U
    int32tofp16 := 0.U
    int16tofp16 := 0.U
    int8tofp16 := 0.U
    in_orR := 0.U
    exp_raw := 0.U
    sign := 0.U

    in_is_fp16 := 0.U
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
    exp_normalized_5 := 0.U
    exp_normalized_8 := 0.U
    sig_normalized_11 := 0.U
    sig_normalized_24 := 0.U
    clz := 0.U
    out_exp_zero_negone := 0.U
    out_exp_16 := 0.U
    out_exp_32 := 0.U
    out_sign := 0.U

    down_exp := 0.U
    nor_sig := 0.U
    nor_roundBit := 0.U
    nor_stickyBit := 0.U
    subnor_in := 0.U
    subnor_roundBit := 0.U
    subnor_stickyBit := 0.U
    may_be_subnor := 0.U
    special_case := 0.U
    nor_of_exp := 0.U
    nor_of_sig := 0.U
    in_isnan_neg := 0.U
    in_sigNotZero := 0.U
    iv := 0.U
    dz := 0.U
    nor_exp := 0.U
    nor_sig_24 := 0.U
    subnor_exp := 0.U
    subnor_sig := 0.U
    isNaN := 0.U
    expIsOnes := 0.U
    expIsZero := 0.U
    isZero := 0.U
    isSubnormal := 0.U
    sigNotZero := 0.U

    in_is_fp32 := (is_sew_32 & is_single | is_sew_16 & is_narrow).asBool

    fp32toint32 := in_is_fp32 && is_single
    fp32toint16 := in_is_fp32 && is_narrow
    fp16toint32 := !in_is_fp32 && is_widen
    fp16toint16 := !in_is_fp32 && is_single
    fp16toint8  := !in_is_fp32 && is_narrow


    val src = Mux(in_is_fp32, io.src,
      Cat(io.src.tail(16).head(1),0.U(3.W), io.src.tail(17).head(f16.expWidth), io.src.tail(f16.expWidth+17), 0.U(13.W)))

    val in = VectorFloat.fromUInt(src, f32.expWidth, f32.precision)
    val expNotZero = Mux1H(
      Seq(fp32toint32 || fp32toint16,
        fp16toint32 || fp16toint16 || fp16toint8),
      Seq(in.decode.expNotZero,
        src.tail(4).head(f16.expWidth).orR)
    )
    raw_in := RawVectorFloat.fromVFP(in, Some(expNotZero)).asUInt

    val max_int_exp = Mux1H(
      Seq(fp32toint32,
        fp32toint16,
        fp16toint32,
        fp16toint16,
        fp16toint8,
        ),
      Seq(VectorFloat.expBias(f32.expWidth).U +& 31.U,
        VectorFloat.expBias(f32.expWidth).U +& 15.U,
        VectorFloat.expBias(f16.expWidth).U +& 31.U,
        VectorFloat.expBias(f16.expWidth).U +& 15.U,
        VectorFloat.expBias(f16.expWidth).U +& 7.U
      )
    )

    exp_of := raw_in.tail(1).head(8) > max_int_exp || (!in_is_fp32 && src.tail(4).head(f16.expWidth).andR)

    // left
    val lpath_shamt = raw_in.tail(1).head(8) - Mux1H(
      Seq(fp32toint32,
        fp16toint16 || fp16toint32
        ),
      Seq((VectorFloat.expBias(f32.expWidth) + f32.fracWidth).U,
        (VectorFloat.expBias(f16.expWidth) + f16.fracWidth).U)
    )
    val lpath_max_shamt = Mux1H(
      Seq(fp32toint32,
        fp16toint16,
        fp16toint32),
      Seq((31 - f32.fracWidth).U,
        (15 - f16.fracWidth).U,
        (31 - f16.fracWidth).U)
    )
    lpath_sig_shifted := Mux1H(
      Seq(fp32toint32,
        fp16toint16 || fp16toint32),
      Seq((raw_in.tail(9) << lpath_shamt(lpath_max_shamt.getWidth - 1, 0))(31, 0),
      (raw_in.tail(9) << lpath_shamt(lpath_max_shamt.getWidth - 1, 0))(31, 13))
    )

    lpath_iv := (fp32toint32 || fp16toint16 || fp16toint32) && !is_signed_int && raw_in.head(1).asBool
    val lpath_may_of = (fp32toint32 || fp16toint16 || fp16toint32) && is_signed_int && (raw_in.tail(1).head(8) === max_int_exp)
    val lpath_pos_of = (fp32toint32 || fp16toint16 || fp16toint32) && lpath_may_of && !raw_in.head(1)
    val lpath_neg_of = (fp32toint32 || fp16toint16 || fp16toint32) && lpath_may_of && raw_in.head(1).asBool && raw_in.tail(9).orR
    lpath_of := lpath_pos_of || lpath_neg_of

    // right
    val rpath_shamt = Mux1H(
      Seq(fp32toint32,
        fp32toint16,
        fp16toint16 || fp16toint32,
        fp16toint8),
      Seq((VectorFloat.expBias(f32.expWidth) + f32.fracWidth).U,
        (VectorFloat.expBias(f32.expWidth) + 15).U,
        (VectorFloat.expBias(f16.expWidth) + f16.fracWidth).U,
        (VectorFloat.expBias(f16.expWidth) + 8 - 1).U)
    ) - raw_in.tail(1).head(8)

    val (rpath_sig_shifted, rpath_sticky) = ShiftRightJam(Cat(raw_in.tail(9),0.U), rpath_shamt)

    rpath_sig_shifted0 := rpath_sig_shifted

    round_in := Mux1H(
      Seq(fp32toint32,
        fp32toint16,
        fp16toint16 || fp16toint32,
        fp16toint8),
      Seq(rpath_sig_shifted.head(f32.precision),
        rpath_sig_shifted.head(16),
        rpath_sig_shifted.head(f16.precision),
        rpath_sig_shifted.head(8))
    )
    round_roundIn := Mux1H(
      Seq(fp32toint32,
        fp32toint16,
        fp16toint16 || fp16toint32,
        fp16toint8),
      Seq(rpath_sig_shifted.tail(f32.precision).head(1),
        rpath_sig_shifted.tail(16).head(1),
        rpath_sig_shifted.tail(f16.precision).head(1),
        rpath_sig_shifted.tail(8).head(1))
    )
    round_stickyIn := Mux1H(
      Seq(fp32toint32,
        fp32toint16,
        fp16toint16 || fp16toint32,
        fp16toint8),
      Seq(rpath_sticky,
        rpath_sticky || rpath_sig_shifted.tail(17).orR,
        rpath_sticky || rpath_sig_shifted.tail(12).orR,
        rpath_sticky || rpath_sig_shifted.tail(9).orR)
    )
    round_signIn := raw_in.head(1).asBool

    sel_lpath := raw_in.tail(1).head(8) >= Mux1H(
      Seq(in_is_fp32,
        !in_is_fp32),
      Seq((VectorFloat.expBias(f32.expWidth) + f32.fracWidth).U,
        (VectorFloat.expBias(f16.expWidth) + f16.fracWidth).U)
    )

    sign_or_nan := Mux1H(
      Seq(fp32toint32 || fp32toint16,
        fp16toint32 || fp16toint16 || fp16toint8),
      Seq(in.decode.isNaN,
        src.tail(4).head(f16.expWidth).andR && src.tail(9).orR)) | !raw_in.head(1).asBool


  }
    .elsewhen(is_int2fp) {

    fp32toint32 := 0.U
    fp32toint16 := 0.U
    fp16toint32 := 0.U
    fp16toint16 := 0.U
    fp16toint8 := 0.U
    raw_in := 0.U
    rpath_sig_shifted0 := 0.U
    sel_lpath := 0.U
    exp_of := 0.U
    lpath_of := 0.U
    lpath_iv := 0.U
    lpath_sig_shifted := 0.U
    sign_or_nan := 0.U

    in_is_fp16 := 0.U
    in_is_fp32 := 0.U
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
    exp_normalized_5 := 0.U
    exp_normalized_8 := 0.U
    sig_normalized_11 := 0.U
    sig_normalized_24 := 0.U
    clz := 0.U
    out_exp_zero_negone := 0.U
    out_exp_16 := 0.U
    out_exp_32 := 0.U
    out_sign := 0.U

    down_exp := 0.U
    nor_sig := 0.U
    nor_roundBit := 0.U
    nor_stickyBit := 0.U
    subnor_in := 0.U
    subnor_roundBit := 0.U
    subnor_stickyBit := 0.U
    may_be_subnor := 0.U
    special_case := 0.U
    nor_of_exp := 0.U
    nor_of_sig := 0.U
    in_isnan_neg := 0.U
    in_sigNotZero := 0.U
    iv := 0.U
    dz := 0.U
    nor_exp := 0.U
    nor_sig_24 := 0.U
    subnor_exp := 0.U
    subnor_sig := 0.U
    isNaN := 0.U
    expIsOnes := 0.U
    expIsZero := 0.U
    isZero := 0.U
    isSubnormal := 0.U
    sigNotZero := 0.U
    /**
     * inttofp
     * ui32 -> fp32
     * i32 ->
     * ui16 ->
     * i16 ->
     *
     * ui32 -> fp16
     * i32 ->
     * ui16 ->
     * i16 ->
     * ui8  ->
     * i8  ->
     *
     */
    out_is_fp32 := (is_sew_32 && is_single) || (is_sew_16 && is_widen)
    int32tofp32 := out_is_fp32 && is_single
    int16tofp32 := out_is_fp32 && is_widen
    int32tofp16 := !out_is_fp32 && is_narrow
    int16tofp16 := !out_is_fp32 && is_single
    int8tofp16 := !out_is_fp32 && is_widen

    sign := is_signed_int && Mux1H(
      Seq(int32tofp32 || int32tofp16,
        int16tofp32 || int16tofp16,
        int8tofp16),
      Seq(io.src.head(1).asBool,
        io.src.tail(16).head(1).asBool,
        io.src.tail(24).head(1).asBool)
    )
    val in_sext = Mux1H(
      Seq(int32tofp32 || int32tofp16,
        int16tofp32 || int16tofp16,
        int8tofp16),
      Seq(io.src,
          Cat(Fill(16, io.src(15)), io.src(15,0)),
        Cat(Fill(24, io.src(7)), io.src(7,0)))
    )
    val in = Mux1H(
      Seq(is_signed_int && (int16tofp32 || int8tofp16 || int16tofp16),
        out_is_fp32 && !(is_signed_int && int16tofp32) || !out_is_fp32 && !(is_signed_int && (int8tofp16 || int16tofp16))),
      Seq(in_sext,
        io.src)
    )
    in_orR := in.orR
    val in_abs = Mux(sign, (~in).asUInt + 1.U, in)
    exp_of := int32tofp16 && in_abs.head(16).orR

    val lzc = Mux1H(
      Seq(out_is_fp32 || exp_of,
        int32tofp16 && !in_abs.head(16).orR || int16tofp16 || int8tofp16,
      ),
      Seq(CLZ(in_abs),
        CLZ(in_abs(15,0)))
    )

    val in_shift = Mux1H(
      Seq(out_is_fp32 || exp_of,
        int32tofp16 && !in_abs.head(16).orR || int16tofp16 || int8tofp16
      ),
      Seq((in_abs << lzc)(30, 0),
        Cat((in_abs << lzc)(14,0), 0.U(16.W)))
    )
    exp_raw := Mux1H(
      Seq(out_is_fp32,
        !out_is_fp32),
      Seq(VectorFloat.expBias(f32.expWidth).asUInt +& 31.U - lzc,
        VectorFloat.expBias(f16.expWidth).asUInt +& 15.U - lzc)
    )

    round_in := Mux1H(
      Seq(int16tofp32,
        int32tofp32,
        int8tofp16,
        int16tofp16 || int32tofp16),
      Seq(in_shift.head(16),
        in_shift.head(23),
        in_shift.head(8),
        in_shift.head(10))
    )
    round_roundIn := Mux1H(
      Seq(int16tofp32,
        int32tofp32,
        int8tofp16,
        int16tofp16 || int32tofp16),
      Seq(in_shift.tail(16).head(1),
        in_shift.tail(23).head(1),
        in_shift.tail(8).head(1),
        in_shift.tail(10).head(1))
    )
    round_stickyIn := Mux1H(
      Seq(int16tofp32,
        int32tofp32,
        int8tofp16,
        int16tofp16 || int32tofp16),
      Seq(in_shift.tail(16).orR,
        in_shift.tail(f32.precision).orR,
        in_shift.tail(8).orR,
        in_shift.tail(f16.precision).orR)
    )
    round_signIn := sign


  }
    .elsewhen(is_vfr) {
    round_in := 0.U
    round_roundIn := false.B
    round_stickyIn := false.B
    round_signIn := false.B

    fp32toint32 := 0.U
    fp32toint16 := 0.U
    fp16toint32 := 0.U
    fp16toint16 := 0.U
    fp16toint8 := 0.U
    raw_in := 0.U
    rpath_sig_shifted0 := 0.U
    sel_lpath := 0.U
    exp_of := 0.U
    lpath_of := 0.U
    lpath_iv := 0.U
    lpath_sig_shifted := 0.U
    sign_or_nan := 0.U

    out_is_fp32 := 0.U
    int32tofp32 := 0.U
    int16tofp32 := 0.U
    int32tofp16 := 0.U
    int16tofp16 := 0.U
    int8tofp16 := 0.U
    in_orR := 0.U
    exp_raw := 0.U
    sign := 0.U

    down_exp := 0.U
    nor_sig := 0.U
    nor_roundBit := 0.U
    nor_stickyBit := 0.U
    subnor_in := 0.U
    subnor_roundBit := 0.U
    subnor_stickyBit := 0.U
    may_be_subnor := 0.U
    special_case := 0.U
    nor_of_exp := 0.U
    nor_of_sig := 0.U
    in_isnan_neg := 0.U
    in_sigNotZero := 0.U
    iv := 0.U
    dz := 0.U
    nor_exp := 0.U
    nor_sig_24 := 0.U
    subnor_exp := 0.U
    subnor_sig := 0.U
    isNaN := 0.U
    expIsOnes := 0.U
    expIsZero := 0.U
    isZero := 0.U
    isSubnormal := 0.U
    sigNotZero := 0.U


    in_is_fp16 := is_sew_16
    in_is_fp32 := is_sew_32


    when(in_is_fp16) {
      out_exp_32 := 0.U
      exp_normalized_8 := 0.U
      sig_normalized_24 := 0.U

      val in = io.src(15, 0)
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
      exp_normalized_5 :=
        Mux(is_vfrsqrt7,
          Mux(is_poszero_posinf,
            Mux(is_normal, exp, Cat(Fill(f16.expWidth - zero_minus_lzc.getWidth, zero_minus_lzc.head(1)), zero_minus_lzc)), 0.U).asUInt,
          Mux(is_normal, exp, Cat(Fill(f16.expWidth - zero_minus_lzc.getWidth, zero_minus_lzc.head(1)), zero_minus_lzc))).asUInt

      sig_normalized_11 := Mux(is_vfrsqrt7,
        Mux(is_poszero_posinf, Mux(is_normal, Cat(0.U, sig), (sig << 1.U).asUInt).asUInt, 0.U),
        Mux(is_normal, Cat(0.U, sig), (sig << 1.U).asUInt))

      clz := CLZ(sig_normalized_11)

      val out_exp_normalized = Mux(is_vfrec7, 29.U - exp_normalized_5, 0.U).asUInt // 2 * bias - 1 - exp_nor
      out_exp_zero_negone := is_vfrec7 && (!out_exp_normalized.orR || out_exp_normalized.andR)

      out_exp_16 := Mux(is_vfrsqrt7,
        Mux(is_normal, (44.U - exp) >> 1, (44.U + CLZ(sig)) >> 1), // if normal (3 * bias - 1 - exp) >> 1 else (3 * bias -1 + CLZ) >>1
        Mux(out_exp_zero_negone, 0.U, out_exp_normalized))

      out_sign := is_poszero_posinf & sign

    }.otherwise {
      exp_normalized_5 := 0.U
      out_exp_16 := 0.U
      sig_normalized_11 := 0.U
      val in = io.src
      sign := in.head(1).asBool
      val exp = in.tail(1).head(f32.expWidth)
      val sig = in.tail(1+f32.expWidth)

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
      is_neg2_bplus1_b := sign & (exp === 254.U)
      is_neg2_b_bminus1 := sign & (exp === 253.U)
      is_neg2_negbminus1_negzero := sign & (sig.head(2) === "b00".U) & is_subnormal & sig.orR
      is_pos2_poszero_negbminus1 := !sign & (sig.head(2) === "b00".U) & is_subnormal & sig.orR
      is_pos2_bminus1_b := !sign & (exp === 253.U)
      is_pos2_b_bplus1 := !sign & (exp === 254.U)


      val zero_minus_lzc = 0.U - CLZ(sig) // 0 - count leading zero
      exp_normalized_8 := Mux(is_vfrsqrt7,
        Mux(is_poszero_posinf, Mux(is_normal, exp, Cat(Fill(f32.expWidth - zero_minus_lzc.getWidth, zero_minus_lzc.head(1)), zero_minus_lzc)), 0.U),
        Mux(is_normal, exp, Cat(Fill(f32.expWidth - zero_minus_lzc.getWidth, zero_minus_lzc.head(1)), zero_minus_lzc)))

      sig_normalized_24 := Mux(is_vfrsqrt7,
        Mux(is_poszero_posinf, Mux(is_normal, Cat(0.U, sig), (sig << 1.U).asUInt), 0.U),
        Mux(is_normal, Cat(0.U, sig), (sig << 1.U).asUInt))

      clz := CLZ(sig_normalized_24)

      val out_exp_normalized = Mux(is_vfrec7, 253.U - exp_normalized_8, 0.U) // 2 * bias - 1 - exp_nor
      out_exp_zero_negone := is_vfrec7 && (!out_exp_normalized.orR || out_exp_normalized.andR)

      out_exp_32 := Mux(is_vfrsqrt7,
        Mux(is_normal, (380.U - exp) >> 1, (380.U + CLZ(sig)) >> 1), // if normal (3 * bias - 1 - exp) >> 1 else (3 * bias -1 + CLZ) >>1
        Mux(out_exp_zero_negone, 0.U, out_exp_normalized))

      out_sign := is_poszero_posinf & sign
    }
  }
    .elsewhen(is_fp2fp) {
    /**
     * fptofp
     * fp32 -> fp16 
     * fp16 -> fp32 
     */
    when(is_narrow) {
      round_in := 0.U
      round_roundIn := false.B
      round_stickyIn := false.B
      round_signIn := false.B

      fp32toint32 := 0.U
      fp32toint16 := 0.U
      fp16toint32 := 0.U
      fp16toint16 := 0.U
      fp16toint8 := 0.U
      raw_in := 0.U
      rpath_sig_shifted0 := 0.U
      sel_lpath := 0.U
      exp_of := 0.U
      lpath_of := 0.U
      lpath_iv := 0.U
      lpath_sig_shifted := 0.U
      sign_or_nan := 0.U

      out_is_fp32 := 0.U
      int32tofp32 := 0.U
      int16tofp32 := 0.U
      int32tofp16 := 0.U
      int16tofp16 := 0.U
      int8tofp16 := 0.U
      in_orR := 0.U
      exp_raw := 0.U
      sign := 0.U

      in_is_fp16 := 0.U
      in_is_fp32 := 0.U
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
      exp_normalized_5 := 0.U
      exp_normalized_8 := 0.U
      sig_normalized_11 := 0.U
      sig_normalized_24 := 0.U
      clz := 0.U
      out_exp_zero_negone := 0.U
      out_exp_16 := 0.U
      out_exp_32 := 0.U
      out_sign := 0.U



      nor_exp := 0.U
      nor_sig_24 := 0.U
      subnor_exp := 0.U
      subnor_sig := 0.U
      isNaN := 0.U
      expIsOnes := 0.U
      expIsZero := 0.U
      isZero := 0.U
      isSubnormal := 0.U
      sigNotZero := 0.U

      val exp_delta = VectorFloat.expBias(f32.expWidth) - VectorFloat.expBias(f16.expWidth)

      val in = VectorFloat.fromUInt(io.src, f32.expWidth, f32.precision)
      val down_exp_s = in.exp.zext - exp_delta.S
      down_exp := down_exp_s.asUInt

      round_signIn := in.sign

      // normal
      nor_sig := in.sig.head(f16.precision - 1)
      nor_roundBit := in.sig.tail(f16.precision - 1).head(1).asBool
      nor_stickyBit := in.sig.tail(f16.precision).orR



      // subnormal
      val shamt = (exp_delta + 1).U(f32.expWidth.W) - in.exp
      val (subnor_sig0, shift_sticky) = ShiftRightJam(Cat(in.decode.expNotZero, in.sig.head(f16.precision)), shamt)
      subnor_sig := subnor_sig0
      subnor_in := subnor_sig.tail(1).head(f16.precision - 1)
      subnor_roundBit := subnor_sig(0)
      subnor_stickyBit := shift_sticky | nor_stickyBit

      may_be_subnor := down_exp.asSInt < 1.S

      val rmin = io.rm === RTZ || (io.rm === RDN & !in.sign) || (io.rm === RUP && in.sign)

      nor_of_exp := Mux(rmin | io.opType(0).asBool, // ROD
        VectorFloat.maxNormExp(f16.expWidth).U(f16.expWidth.W),
        (VectorFloat.maxNormExp(f16.expWidth) + 1).U(f16.expWidth.W))

      // ROD
      nor_of_sig := Mux(rmin | io.opType(0).asBool, ~0.U((f16.precision - 1).W), 0.U((f16.precision - 1).W))

      special_case := in.decode.expIsOnes

      iv := in.decode.isSNaN
      dz := false.B

      in_isnan_neg := !in.decode.isNaN && in.sign
      in_sigNotZero := in.decode.sigNotZero


    }
      .elsewhen(is_widen) {
      round_in := 0.U
      round_roundIn := false.B
      round_stickyIn := false.B
      round_signIn := false.B

      fp32toint32 := 0.U
      fp32toint16 := 0.U
      fp16toint32 := 0.U
      fp16toint16 := 0.U
      fp16toint8 := 0.U
      raw_in := 0.U
      rpath_sig_shifted0 := 0.U
      sel_lpath := 0.U
      exp_of := 0.U
      lpath_of := 0.U
      lpath_iv := 0.U
      lpath_sig_shifted := 0.U
      sign_or_nan := 0.U

      out_is_fp32 := 0.U
      int32tofp32 := 0.U
      int16tofp32 := 0.U
      int32tofp16 := 0.U
      int16tofp16 := 0.U
      int8tofp16 := 0.U
      in_orR := 0.U
      exp_raw := 0.U
      sign := 0.U

      in_is_fp16 := 0.U
      in_is_fp32 := 0.U
      is_normal := 0.U
      is_inf := 0.U
      is_nan := 0.U
      is_neginf := 0.U
      is_neginf_negzero := 0.U
      is_negzero := 0.U
      is_poszero := 0.U
      is_neg2_bplus1_b := 0.U
      is_neg2_b_bminus1 := 0.U
      is_neg2_negbminus1_negzero := 0.U
      is_pos2_poszero_negbminus1 := 0.U
      is_pos2_bminus1_b := 0.U
      is_pos2_b_bplus1 := 0.U
      exp_normalized_5 := 0.U
      exp_normalized_8 := 0.U
      sig_normalized_11 := 0.U
      sig_normalized_24 := 0.U
      clz := 0.U
      out_exp_zero_negone := 0.U
      out_exp_16 := 0.U
      out_exp_32 := 0.U
      out_sign := 0.U

      down_exp := 0.U
      nor_sig := 0.U
      nor_roundBit := 0.U
      nor_stickyBit := 0.U
      subnor_in := 0.U
      subnor_roundBit := 0.U
      subnor_stickyBit := 0.U
      may_be_subnor := 0.U
      special_case := 0.U
      nor_of_exp := 0.U
      nor_of_sig := 0.U
      in_isnan_neg := 0.U
      in_sigNotZero := 0.U
      iv := 0.U
      dz := 0.U



      val in = VectorFloat.fromUInt(io.src(15,0), f16.expWidth, f16.precision)
      val exp_delta = VectorFloat.expBias(f32.expWidth) - VectorFloat.expBias(f16.expWidth)
      nor_sig_24 := in.sig
      nor_exp := exp_delta.U(f32.expWidth.W) + in.exp

      val subnor_shamt = CLZ(in.sig)
      subnor_sig := Cat((in.sig << subnor_shamt)(f16.precision - 3, 0), 0.U(1.W))
      subnor_exp := exp_delta.U(f32.expWidth.W) - subnor_shamt


      isNaN := in.decode.isNaN
      expIsOnes := in.decode.expIsOnes
      expIsZero := in.decode.expIsZero
      isZero := in.decode.isZero
      isSubnormal := in.decode.isSubnormal
      sigNotZero := in.decode.sigNotZero
      is_snan := in.decode.isSNaN

      sign := in.sign

    }.otherwise {
      round_in := 0.U
      round_roundIn := false.B
      round_stickyIn := false.B
      round_signIn := false.B

      fp32toint32 := 0.U
      fp32toint16 := 0.U
      fp16toint32 := 0.U
      fp16toint16 := 0.U
      fp16toint8 := 0.U
      raw_in := 0.U
      rpath_sig_shifted0 := 0.U
      sel_lpath := 0.U
      exp_of := 0.U
      lpath_of := 0.U
      lpath_iv := 0.U
      lpath_sig_shifted := 0.U
      sign_or_nan := 0.U

      out_is_fp32 := 0.U
      int32tofp32 := 0.U
      int16tofp32 := 0.U
      int32tofp16 := 0.U
      int16tofp16 := 0.U
      int8tofp16 := 0.U
      in_orR := 0.U
      exp_raw := 0.U
      sign := 0.U

      in_is_fp16 := 0.U
      in_is_fp32 := 0.U
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
      exp_normalized_5 := 0.U
      exp_normalized_8 := 0.U
      sig_normalized_11 := 0.U
      sig_normalized_24 := 0.U
      clz := 0.U
      out_exp_zero_negone := 0.U
      out_exp_16 := 0.U
      out_exp_32 := 0.U
      out_sign := 0.U

      down_exp := 0.U
      nor_sig := 0.U
      nor_roundBit := 0.U
      nor_stickyBit := 0.U
      subnor_in := 0.U
      subnor_roundBit := 0.U
      subnor_stickyBit := 0.U
      may_be_subnor := 0.U
      special_case := 0.U
      nor_of_exp := 0.U
      nor_of_sig := 0.U
      in_isnan_neg := 0.U
      in_sigNotZero := 0.U
      iv := 0.U
      dz := 0.U
      nor_exp := 0.U
      nor_sig_24 := 0.U
      subnor_exp := 0.U
      subnor_sig := 0.U
      isNaN := 0.U
      expIsOnes := 0.U
      expIsZero := 0.U
      isZero := 0.U
      isSubnormal := 0.U
      sigNotZero := 0.U
    }
  }
    .otherwise {
    round_in := 0.U
    round_roundIn := false.B
    round_stickyIn := false.B
    round_signIn := false.B

    fp32toint32 := 0.U
    fp32toint16 := 0.U
    fp16toint32 := 0.U
    fp16toint16 := 0.U
    fp16toint8 := 0.U
    raw_in := 0.U
    rpath_sig_shifted0 := 0.U
    sel_lpath := 0.U
    exp_of := 0.U
    lpath_of := 0.U
    lpath_iv := 0.U
    lpath_sig_shifted := 0.U
    sign_or_nan := 0.U

    out_is_fp32 := 0.U
    int32tofp32 := 0.U
    int16tofp32 := 0.U
    int32tofp16 := 0.U
    int16tofp16 := 0.U
    int8tofp16 := 0.U
    in_orR := 0.U
    exp_raw := 0.U
    sign := 0.U

    in_is_fp16 := 0.U
    in_is_fp32 := 0.U
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
    exp_normalized_5 := 0.U
    exp_normalized_8 := 0.U
    sig_normalized_11 := 0.U
    sig_normalized_24 := 0.U
    clz := 0.U
    out_exp_zero_negone := 0.U
    out_exp_16 := 0.U
    out_exp_32 := 0.U
    out_sign := 0.U

    down_exp := 0.U
    nor_sig := 0.U
    nor_roundBit := 0.U
    nor_stickyBit := 0.U
    subnor_in := 0.U
    subnor_roundBit := 0.U
    subnor_stickyBit := 0.U
    may_be_subnor := 0.U
    special_case := 0.U
    nor_of_exp := 0.U
    nor_of_sig := 0.U
    in_isnan_neg := 0.U
    in_sigNotZero := 0.U
    iv := 0.U
    dz := 0.U
    nor_exp := 0.U
    nor_sig_24 := 0.U
    subnor_exp := 0.U
    subnor_sig := 0.U
    isNaN := 0.U
    expIsOnes := 0.U
    expIsZero := 0.U
    isZero := 0.U
    isSubnormal := 0.U
    sigNotZero := 0.U

  }

  when(RegNext(is_fp2int)) {
    val rpath_rounder = Module(new RoundingUnit(f32.precision))
    rpath_rounder.io.in := round_in_reg0
    rpath_rounder.io.roundIn := round_roundIn_reg0
    rpath_rounder.io.stickyIn := round_stickyIn_reg0
    rpath_rounder.io.signIn := round_signIn_reg0
    rpath_rounder.io.rm := rm_reg0

    val out_reg0 = Mux(rpath_rounder.io.r_up, rpath_rounder.io.in + 1.U, rpath_rounder.io.in)
    val cout_reg0 = rpath_rounder.io.r_up && Mux1H(
      Seq(fp32toint32_reg0 || fp16toint32_reg0,
        fp32toint16_reg0 || fp16toint16_reg0,
        fp16toint8_reg0),
      Seq(rpath_rounder.io.in.andR,
        rpath_rounder.io.in.tail(9).andR,
        rpath_rounder.io.in.tail(17).andR)
    )

    val rpath_sig_reg0 = Mux1H(
      Seq(fp32toint32_reg0,
        fp32toint16_reg0,
        fp16toint16_reg0 || fp16toint32_reg0,
        fp16toint8_reg0),
      Seq(Cat(0.U((32 - f32.precision - 1).W), cout_reg0, out_reg0),
        Cat(0.U(8.W), out_reg0),
        Cat(0.U((32 - f16.precision - 1).W), cout_reg0, out_reg0(10, 0)),
        Cat(0.U(21.W), out_reg0(10, 0)))
    )

    val rpath_ix_reg0 = rpath_rounder.io.inexact || (fp16toint8_reg0 && rpath_sig_shifted_reg0(7,0).orR)
    val rpath_iv_reg0 = !is_signed_int_reg0 && raw_in_reg0.head(1).asBool && rpath_sig_reg0.orR

    val rpath_pos_of_reg0 = !raw_in_reg0.head(1).asBool &&
      Mux(is_signed_int_reg0,
        Mux1H(
          Seq(fp32toint16_reg0,
            fp16toint8_reg0),
          Seq((raw_in_reg0.tail(1).head(8) === 142.U) || (raw_in_reg0.tail(1).head(8) === 141.U) && cout_reg0,
            (raw_in_reg0.tail(1).head(8) === 22.U) || (raw_in_reg0.tail(1).head(8) === 21.U) && cout_reg0)
        ),
        Mux1H(
          Seq(fp32toint16_reg0,
            fp16toint8_reg0),
          Seq((raw_in_reg0.tail(1).head(8) === 142.U) && cout_reg0,
            (raw_in_reg0.tail(1).head(8) === 22.U) && cout_reg0)
        )
      )

    val rpath_neg_of_reg0 = raw_in_reg0.head(1).asBool && Mux1H(
      Seq(fp32toint16_reg0,
        fp16toint8_reg0),
      Seq(raw_in_reg0.tail(1).head(8) === 142.U && (rpath_rounder.io.in.tail(8).orR || rpath_rounder.io.r_up),
        raw_in_reg0.tail(1).head(8) === 22.U && (rpath_rounder.io.in.tail(17).orR || rpath_rounder.io.r_up))
    )
    val rpath_of_reg0 = Mux1H(
      Seq(fp32toint16_reg0 || fp16toint8_reg0,
        fp32toint32_reg0 || fp16toint16_reg0 || fp16toint32_reg0),
      Seq(rpath_neg_of_reg0 || rpath_pos_of_reg0,
        cout_reg0)
    )

    val of = exp_of_reg0 || sel_lpath_reg0 && lpath_of_reg0 || !sel_lpath_reg0 && rpath_of_reg0
    val iv = of || sel_lpath_reg0 && lpath_iv_reg0 || !sel_lpath_reg0 && rpath_iv_reg0
    val ix = !iv && !sel_lpath_reg0 && rpath_ix_reg0

    val int_abs = Mux(sel_lpath_reg0, lpath_sig_shifted_reg0, rpath_sig_reg0)
    val sign = raw_in_reg0.head(1).asBool && is_signed_int_reg0
    val raw_int = Mux1H(
      Seq(fp32toint16_reg0 || fp16toint16_reg0,
        fp32toint32_reg0 || fp16toint32_reg0,
        fp16toint8_reg0),
      Seq(int_abs.tail(16),
        int_abs,
        int_abs.tail(24))
    )
    val int = Mux(sign, -raw_int, raw_int)

    val max_int = Mux1H(
      Seq(fp32toint32_reg0 || fp16toint32_reg0,
        fp32toint16_reg0 || fp16toint16_reg0,
        fp16toint8_reg0),
      Seq(Cat(!is_signed_int_reg0, ~0.U(31.W)),
        Cat(!is_signed_int_reg0, ~0.U(15.W)),
        Cat(!is_signed_int_reg0, ~0.U(7.W)))
    )
    val min_int = Mux1H(
      Seq(fp32toint32_reg0 || fp16toint32_reg0,
        fp32toint16_reg0 || fp16toint16_reg0,
        fp16toint8_reg0),
      Seq(Cat(is_signed_int_reg0, 0.U(31.W)),
        Cat(is_signed_int_reg0, 0.U(15.W)),
        Cat(is_signed_int_reg0, 0.U(7.W)))
    )

    result0 := Mux(iv, Mux(sign_or_nan_reg0, max_int, min_int), int)
    fflags0 := Cat(iv, false.B, false.B, false.B, ix)
  }
    .elsewhen(RegNext(is_int2fp)) {
    val rounder = Module(new RoundingUnit(f32.fracWidth))
    rounder.io.in := round_in_reg0
    rounder.io.roundIn := round_roundIn_reg0
    rounder.io.stickyIn := round_stickyIn_reg0
    rounder.io.signIn := round_signIn_reg0
    rounder.io.rm := rm_reg0

    val out_reg0 = Mux(rounder.io.r_up, rounder.io.in + 1.U, rounder.io.in)
    val cout_reg0 = rounder.io.r_up && Mux1H(
      Seq(int16tofp32_reg0,
        int32tofp32_reg0,
        int8tofp16_reg0,
        int16tofp16_reg0 || int32tofp16_reg0),
      Seq(rounder.io.in.tail(7).andR,
        rounder.io.in.andR,
        rounder.io.in.tail(15).andR,
        rounder.io.in.tail(13).andR))

    val exp_reg0 = Mux(in_orR_reg0, exp_raw_reg0 + cout_reg0, 0.U)
    val sig_reg0 = out_reg0

    val of = Mux1H(
      Seq(out_is_fp32_reg0,
        int32tofp16_reg0,
        !out_is_fp32_reg0 && !int32tofp16_reg0),
      Seq(exp_reg0 === 255.U,
        exp_of_reg0 || exp_reg0 === 31.U,
        exp_reg0 === 31.U)
    )
    val ix = Mux1H(
      Seq(out_is_fp32_reg0 || int16tofp16_reg0 || int8tofp16_reg0,
        int32tofp16_reg0),
      Seq(rounder.io.inexact,
        exp_of_reg0 || rounder.io.inexact)
    )

    val result_fp = Cat(sign_reg0, Mux(RoundingUnit.is_rmin(rm_reg0, sign_reg0), Cat(Fill(4, 1.U), 0.U, Fill(10, 1.U)), Cat(Fill(5, 1.U), Fill(10, 0.U))))

    result0 := Mux1H(
      Seq(int32tofp32_reg0,
        int16tofp32_reg0,
        int32tofp16_reg0 && of,
        int8tofp16_reg0,
        int32tofp16_reg0 && !of || int16tofp16_reg0),
      Seq(Cat(round_signIn_reg0, exp_reg0, sig_reg0),
        Cat(round_signIn_reg0, exp_reg0, sig_reg0.tail(7), 0.U(7.W)),
        result_fp,
        Cat(round_signIn_reg0, exp_reg0(4, 0), sig_reg0(7, 0), 0.U(2.W)),
        Cat(round_signIn_reg0, exp_reg0(4, 0), sig_reg0(9, 0)))
    )
    fflags0 := Cat(false.B, false.B, of, false.B, ix)
  }
    .elsewhen(RegNext(is_vfr)) {

    val vfrsqrt7Table = Module(new Rsqrt7Table)
    val vfrec7Table = Module(new Rec7Table)
    when(in_is_fp16_reg0){
      val sig_in7 = Mux(is_vfrsqrt7_reg0,
        Cat(exp_normalized0_reg0(0), (sig_normalized0_reg0 << Mux(is_normal_reg0, 0.U, clz_reg0))(9, 4)).asUInt, // vfrsqrt7  Cat(exp_nor(0), sig_nor(9,4))
        (sig_normalized0_reg0 << Mux(is_normal_reg0, 0.U, clz_reg0))(9, 3)).asUInt // vfrec7 sig_nor(9,3)

      vfrsqrt7Table.src := sig_in7
      vfrec7Table.src := sig_in7

      val sig_out7 = Wire(UInt(7.W))
      sig_out7 := Mux(is_vfrsqrt7_reg0, vfrsqrt7Table.out, vfrec7Table.out)

      val out_sig =
        Mux(is_vfrec7_reg0,
          Mux(out_exp_zero_negone_reg0,
            Mux(is_neg2_bplus1_b_reg0 || is_pos2_b_bplus1_reg0,
              Cat(0.U, 1.U, sig_out7, 0.U),
              Mux(is_neg2_b_bminus1_reg0 | is_pos2_bminus1_b_reg0,
                Cat(1.U, sig_out7, 0.U(2.W)),
                Cat(1.U, sig_out7, 0.U(2.W)) >> 1.U)),
            Cat(sig_out7, 0.U(3.W))),
          0.U)

      val fp_result = Wire(UInt(16.W))
      fp_result := Mux1H(
        Seq(is_vfrsqrt7_reg0,
          is_vfrec7_reg0),
        Seq(Cat(out_sign_reg0, out_exp_16_reg0, sig_out7, 0.U(3.W)),
          Cat(sign_reg0, out_exp_16_reg0, out_sig)
        ))

      val result_nan = Cat(0.U, Fill(6, 1.U), 0.U(9.W))
      val result_inf = Cat(Fill(5, 1.U), Fill(10, 0.U))
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

    }.elsewhen(in_is_fp32_reg0) {
      val sig_in7 = Mux(is_vfrsqrt7_reg0,
        Cat(exp_normalized1_reg0(0), (sig_normalized1_reg0 << Mux(is_normal_reg0, 0.U, clz_reg0))(22, 17)),
        (sig_normalized1_reg0 << Mux(is_normal_reg0, 0.U, clz_reg0))(22, 16))

      vfrsqrt7Table.src := sig_in7
      vfrec7Table.src := sig_in7

      val sig_out7 = Wire(UInt(7.W))
      sig_out7 := Mux(is_vfrsqrt7_reg0, vfrsqrt7Table.out, vfrec7Table.out)


      val out_sig =
        Mux(is_vfrec7_reg0,
          Mux(out_exp_zero_negone_reg0,
            Mux(is_neg2_bplus1_b_reg0 || is_pos2_b_bplus1_reg0,
              Cat(0.U, 1.U, sig_out7, 0.U(14.W)),
              Mux(is_neg2_b_bminus1_reg0 || is_pos2_bminus1_b_reg0,
                Cat(1.U, sig_out7, 0.U(15.W)),
                Cat(1.U, sig_out7, 0.U(15.W)) >> 1.U)),
            Cat(sig_out7, 0.U(16.W))),
          0.U)


      val fp_result = Wire(UInt(32.W))
      fp_result := Mux1H(
        Seq(is_vfrsqrt7_reg0,
          is_vfrec7_reg0),
        Seq(Cat(out_sign_reg0, out_exp_32_reg0, sig_out7, 0.U(16.W)),
          Cat(sign_reg0, out_exp_32_reg0, out_sig)
        ))

      val result_nan = Cat(0.U, Fill(9, 1.U), 0.U(22.W))
      val result_inf = Cat(Fill(8, 1.U), Fill(23, 0.U))
      val result_greatest_fin = Cat(Fill(7, 1.U), 0.U, Fill(23, 1.U))
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
          Mux(is_neginf_reg0, Cat(1.U, 0.U(31.W)), 0.U),
          Mux(is_negzero_reg0, Cat(Fill(9, 1.U), 0.U(23.W)), Cat(0.U(1.W), Fill(8, 1.U), 0.U(23.W))),
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
      vfrsqrt7Table.src := 0.U
      vfrec7Table.src := 0.U
      result0 := 0.U
    }
  }
    .elsewhen(RegNext(is_fp2fp)) {
    when(is_narrow_reg0) {
      val nor_rounder = Module(new RoundingUnit(f16.precision - 1))
      nor_rounder.io.in := nor_sig_reg0
      nor_rounder.io.roundIn := nor_roundBit_reg0
      nor_rounder.io.stickyIn := nor_stickyBit_reg0
      nor_rounder.io.signIn := round_signIn_reg0
      nor_rounder.io.rm := rm_reg0

      val out_reg0 = Mux(nor_rounder.io.r_up, nor_rounder.io.in + 1.U, nor_rounder.io.in)
      val cout_reg0 = nor_rounder.io.r_up && nor_rounder.io.in.andR

      val nor_sig_rounded_reg0 = out_reg0
      val nor_exp_rounded_reg0 = Mux(cout_reg0, down_exp_reg0.asSInt + 1.S, down_exp_reg0.asSInt)
      val nor_of_reg0 =
        Mux(cout_reg0,
          down_exp_reg0.asSInt > (VectorFloat.maxNormExp(f16.expWidth) - 1).S,
          down_exp_reg0.asSInt > VectorFloat.maxNormExp(f16.expWidth).S).asBool

      val exp_uf_reg0 = Mux(cout_reg0, down_exp_reg0.asSInt < 0.S, down_exp_reg0.asSInt < 1.S).asBool
      val nor_ix_reg0 = nor_of_reg0 | nor_rounder.io.inexact

      val subnor_rounder = Module(new RoundingUnit(f16.precision - 1))
      subnor_rounder.io.in := subnor_in_reg0
      subnor_rounder.io.roundIn := subnor_roundBit_reg0
      subnor_rounder.io.stickyIn := subnor_stickyBit_reg0
      subnor_rounder.io.signIn := round_signIn_reg0
      subnor_rounder.io.rm := rm_reg0

      val out1_reg0 = Mux(subnor_rounder.io.r_up, subnor_rounder.io.in + 1.U, subnor_rounder.io.in)
      val cout1_reg0 = subnor_rounder.io.r_up && subnor_rounder.io.in.andR

      val subnor_sig_rounded_reg0 = out1_reg0
      val subnor_exp_rounded_reg0 = Mux(cout1_reg0, 1.U, 0.U)
      val subnor_ix_reg0 = subnor_rounder.io.inexact

      val common_exp = Mux1H(
        Seq(
          !may_be_subnor_reg0 & nor_of_reg0,
          !may_be_subnor_reg0 & !nor_of_reg0,
          may_be_subnor_reg0
        ),
        Seq(
          nor_of_exp_reg0,
          nor_exp_rounded_reg0(f16.expWidth - 1, 0),
          subnor_exp_rounded_reg0
        )
      )

      val common_sig = Mux1H(
        Seq(
          !may_be_subnor_reg0 & nor_of_reg0,
          !may_be_subnor_reg0 & !nor_of_reg0,
          may_be_subnor_reg0
        ),
        Seq(
          nor_of_sig_reg0(9,0),
          nor_sig_rounded_reg0,
          subnor_sig_rounded_reg0
        )
      )

      val of = !special_case_reg0 & nor_of_reg0
      val uf = !special_case_reg0 & may_be_subnor_reg0 & subnor_ix_reg0 & exp_uf_reg0
      val ix = !special_case_reg0 & (!may_be_subnor_reg0 & nor_ix_reg0 | may_be_subnor_reg0 & subnor_ix_reg0)

      result0 := Cat(
        in_isnan_neg_reg0,
        Mux1H(
          Seq(special_case_reg0, !special_case_reg0),
          Seq(~0.U(f16.expWidth.W), common_exp)
        ),
        Mux1H(
          Seq(special_case_reg0, !special_case_reg0),
          Seq(
            Cat(in_sigNotZero_reg0, 0.U((f16.precision - 2).W)),
            common_sig
          )))
      fflags0 := Cat(iv_reg0, dz_reg0, of, uf, ix)

    }
      .elsewhen(is_widen_reg0) {

      result0 := Cat(
        !isNaN_reg0 & sign_reg0,
        Mux1H(
          Seq(
            expIsOnes_reg0,
            isZero_reg0,
            isSubnormal_reg0,
            !expIsOnes_reg0 & !expIsZero_reg0
          ),
          Seq(
            ~0.U(f32.expWidth.W),
            0.U(f32.expWidth.W),
            subnor_exp_reg0,
            nor_exp_reg0
          )
        ),
        Mux1H(
          Seq(
            expIsOnes_reg0,
            expIsZero_reg0,
            !expIsOnes_reg0 & !expIsZero_reg0
          ),
          Seq(
            Cat(sigNotZero_reg0, 0.U((f32.precision - 2).W)),
            Cat(subnor_sig_reg0(9,0), 0.U((f32.precision - f16.precision).W)),
            Cat(nor_sig_24_reg0, 0.U((f32.precision - f16.precision).W))
          )
        )
      )
      fflags0 := Cat(is_snan_reg0, 0.U(4.W))

    }.otherwise {
      result0 := 0.U
    }

  }
    .otherwise {
    result0 := 0.U
  }

  result := result0_reg0
  fflags := fflags0_reg0

  io.result := result
  io.fflags := fflags
   */


  /** cycle0                                                              | cycle1                  |   cycle2
   * fp2int   in(32) raw_in(33)  left,right    ShiftRightJam(25)          | RoundingUnit(11)  adder |
   * int2fp   in(32) in_abs(32)  lzc in_shift  exp_raw                    | RoundingUnit(10)  adder |  -> result & fflags
   * vfr      in(32)             lzc exp_nor   sig_nor  clz out_exp adder | Table                   |
   * fp2fp
   */
  // control path
  val is_sew_8  = io.sew === "b00".U
  val is_sew_16 = io.sew === "b01".U
  val is_sew_32 = io.sew === "b10".U

  val is_single = io.opType.tail(3).head(2) === "b00".U
  val is_widen = io.opType.tail(3).head(2) === "b01".U
  val is_narrow = io.opType.tail(3).head(2) === "b10".U
  val is_single_reg0 = RegNext(is_single)
  val is_widen_reg0 = RegNext(is_widen)
  val is_narrow_reg0 = RegNext(is_narrow)


  val is_vfr = io.opType(5).asBool && (is_sew_16 || is_sew_32)
  val is_fp2int = io.opType.head(2) === "b10".U && (is_sew_32 && is_single || is_sew_16 && is_narrow || is_sew_16 && is_widen || is_sew_16 && is_single || is_sew_8 && is_narrow)
  val is_int2fp = io.opType.head(2) === "b01".U && (is_sew_32 && is_single || is_sew_16 && is_narrow || is_sew_16 && is_widen || is_sew_16 && is_single || is_sew_8 && is_widen)
  val is_fp2fp = io.opType.head(3) === "b110".U && (is_sew_16 && (is_narrow || is_widen))
  val is_vfr_reg0 = RegNext(is_vfr, false.B)
  val is_fp2int_reg0 = RegNext(is_fp2int, false.B)
  val is_int2fp_reg0 = RegNext(is_int2fp, false.B)
  val is_fp2fp_reg0 = RegNext(is_fp2fp, false.B)

  val is_vfrsqrt7 = is_vfr && !io.opType(0).asBool
  val is_vfrec7 = is_vfr && io.opType(0).asBool
  val is_vfrsqrt7_reg0 = RegNext(is_vfrsqrt7, false.B)
  val is_vfrec7_reg0 = RegNext(is_vfrec7, false.B)

  val is_signed_int = io.opType(0)
  val is_signed_int_reg0 = RegNext(is_signed_int, false.B)

  val result = Wire(UInt(32.W))
  val NV, DZ, OF, UF, NX = WireInit(false.B)
  val fflags = WireInit(Cat(NV, DZ, OF, UF, NX))

  val result0 = Wire(UInt(32.W))
  val result0_reg0 = RegNext(result0, 0.U(32.W))
  val fflags0 = WireInit(Cat(NV, DZ, OF, UF, NX))
  val fflags0_reg0 = RegNext(fflags0)

  val round_in = Wire(UInt(24.W))
  val round_roundIn = Wire(Bool())
  val round_stickyIn = Wire(Bool())
  val round_signIn = Wire(Bool())
  val round_in_reg0 = RegNext(round_in, 0.U(24.W))
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

  // fp2fp
  val nor_in = Wire(UInt(10.W))
  val nor_roundBit = Wire(Bool())
  val nor_stickyBit = Wire(Bool())
  val nor_signBit = Wire(Bool())
  val nor_in_reg0 = RegNext(nor_in)
  val nor_roundBit_reg0 = RegNext(nor_roundBit)
  val nor_stickyBit_reg0 = RegNext(nor_stickyBit)
  val nor_signBit_reg0 = RegNext(nor_signBit)
  val subnor_in = Wire(UInt(10.W))
  val subnor_roundBit = Wire(Bool())
  val subnor_stickyBit = Wire(Bool())
  val subnor_signBit = Wire(Bool())
  val subnor_in_reg0 = RegNext(subnor_in)
  val subnor_roundBit_reg0 = RegNext(subnor_roundBit)
  val subnor_stickyBit_reg0 = RegNext(subnor_stickyBit)
  val subnor_signBit_reg0 = RegNext(subnor_signBit)


  // cycle0

  val in_is_fp16 = Wire(Bool())
  val in_is_fp32 = Wire(Bool())
  val fp32toint32 = Wire(Bool())
  val fp32toint16 = Wire(Bool())
  val fp16toint32 = Wire(Bool())
  val fp16toint16 = Wire(Bool())
  val fp16toint8 = Wire(Bool())

  val in_is_fp16_reg0 = RegNext(in_is_fp16)
  val in_is_fp32_reg0 = RegNext(in_is_fp32)
  val fp32toint32_reg0 = RegNext(fp32toint32)
  val fp32toint16_reg0 = RegNext(fp32toint16)
  val fp16toint32_reg0 = RegNext(fp16toint32)
  val fp16toint16_reg0 = RegNext(fp16toint16)
  val fp16toint8_reg0 = RegNext(fp16toint8)

  val out_is_fp16 = Wire(Bool())
  val out_is_fp32 = Wire(Bool())
  val int32tofp32 = Wire(Bool())
  val int16tofp32 = Wire(Bool())
  val int32tofp16 = Wire(Bool())
  val int16tofp16 = Wire(Bool())
  val int8tofp16 = Wire(Bool())
  val out_is_fp16_reg0 = RegNext(out_is_fp16)
  val out_is_fp32_reg0 = RegNext(out_is_fp32)
  val int32tofp32_reg0 = RegNext(int32tofp32)
  val int16tofp32_reg0 = RegNext(int16tofp32)
  val int32tofp16_reg0 = RegNext(int32tofp16)
  val int16tofp16_reg0 = RegNext(int16tofp16)
  val int8tofp16_reg0 = RegNext(int8tofp16)

  in_is_fp16 := Mux1H(
    Seq(is_fp2int,
      is_vfr
    ),
    Seq(is_sew_16 && is_widen || is_sew_16 && is_single || is_sew_8 && is_narrow,
      is_sew_16
    )
  )

  in_is_fp32 := Mux1H(
    Seq(is_fp2int,
      is_vfr),
    Seq(is_sew_32 && is_single || is_sew_16 && is_narrow,
      is_sew_32)
  )
  fp32toint32 := is_fp2int && in_is_fp32 && is_single
  fp32toint16 := is_fp2int && in_is_fp32 && is_narrow
  fp16toint32 := is_fp2int && in_is_fp16 && is_widen
  fp16toint16 := is_fp2int && in_is_fp16 && is_single
  fp16toint8  := is_fp2int && in_is_fp16 && is_narrow

  out_is_fp32 := Mux1H(
    Seq(is_int2fp,
      is_vfr,
      is_fp2fp),
    Seq(is_sew_32 && is_single || is_sew_16 && is_widen,
      is_sew_32,
      is_widen)
  )
  out_is_fp16 := Mux1H(
    Seq(is_int2fp,
      is_vfr,
      is_fp2fp),
    Seq(is_sew_32 && is_narrow || is_sew_16 && is_single || is_sew_8 && is_widen,
      is_sew_16,
      is_narrow)
  )
  int32tofp32 := is_int2fp && out_is_fp32 && is_single
  int16tofp32 := is_int2fp && out_is_fp32 && is_widen
  int32tofp16 := is_int2fp && out_is_fp16 && is_narrow
  int16tofp16 := is_int2fp && out_is_fp16 && is_single
  int8tofp16  := is_int2fp && out_is_fp16 && is_widen

  // in
  val src = Mux1H(
    Seq(in_is_fp32 || (is_vfr && is_sew_32) || (is_fp2fp && is_narrow),
      in_is_fp16 || (is_vfr && is_sew_16) || (is_fp2fp && is_widen)),
    Seq(io.src,
      Cat(io.src.tail(16).head(1), 0.U(3.W), io.src.tail(17).head(f16.expWidth), io.src.tail(f16.expWidth+17), 0.U(13.W)))
  )
  val fp_in = VectorFloat.fromUInt(src, f32.expWidth, f32.precision)
  val exp_not_zero = Mux1H(
    Seq(fp32toint32 || fp32toint16,
      fp16toint32 || fp16toint16 || fp16toint8),
    Seq(fp_in.decode.expNotZero,
      src.tail(4).head(f16.expWidth).orR)
  )
  val fp2int_in = Wire(UInt(33.W))
  fp2int_in := RawVectorFloat.fromVFP(fp_in, Some(exp_not_zero)).asUInt

  val in_sext = Wire(UInt(33.W))
  in_sext := Mux1H(
    Seq(!is_signed_int && (int32tofp32 || int32tofp16),
      is_signed_int && (int32tofp32 || int32tofp16),
      !is_signed_int && (int16tofp32 || int16tofp16),
      is_signed_int && (int16tofp32 || int16tofp16),
      !is_signed_int && int8tofp16,
      is_signed_int && int8tofp16
    ),
    Seq(Cat(0.U, io.src),
      Cat(io.src(31), io.src),
      Cat(0.U(17.W), io.src(15,0)),
      Cat(Fill(17, io.src(15)), io.src(15,0)),
      Cat(0.U(25.W), io.src(7,0)),
      Cat(Fill(25, io.src(7)), io.src(7,0))
    )
  )
  val in_orR = Wire(Bool())
  val in_orR_reg0 = RegNext(in_orR, false.B)
  in_orR := in_sext.orR

  val in = Wire(UInt(33.W))
  val in_reg0 = RegNext(in, 0.U(33.W))
  in := Mux1H(
    Seq(is_fp2int,
      is_int2fp,
      is_vfr),
    Seq(fp2int_in,
      in_sext,
      Cat(0.U, src))
  )
  val sign = Wire(Bool())
  val sign_reg0 = RegNext(sign, false.B)
  sign := Mux1H(
    Seq(is_fp2int || is_int2fp,
      is_vfr,
      is_fp2fp && is_widen),
    Seq(in.head(1).asBool,
      in.tail(1).head(1).asBool,
      fp_in.sign)
  )

  val exp = Mux1H(
    Seq(is_fp2int,
      is_vfr),
    Seq(in.tail(1).head(f32.expWidth),
      in.tail(2).head(f32.expWidth))
  )
  val sig = Mux1H(
    Seq(is_fp2int,
      is_vfr),
    Seq(in.tail(9),
      in.tail(10))
  )

  // fp2int
  // left
  val max_int_exp = Mux1H(
    Seq(fp32toint32,
      fp32toint16,
      fp16toint32,
      fp16toint16,
      fp16toint8,
    ),
    Seq(VectorFloat.expBias(f32.expWidth).U +& 31.U,
      VectorFloat.expBias(f32.expWidth).U +& 15.U,
      VectorFloat.expBias(f16.expWidth).U +& 31.U,
      VectorFloat.expBias(f16.expWidth).U +& 15.U,
      VectorFloat.expBias(f16.expWidth).U +& 7.U
    )
  )

  val exp_of = Wire(Bool())
  val exp_of_reg0 = RegNext(exp_of, false.B)
  exp_of := Mux1H(
    Seq(is_fp2int && in_is_fp32,
      is_fp2int && in_is_fp16),
    Seq(exp > max_int_exp,
      exp > max_int_exp || src.tail(4).head(f16.expWidth).andR)
  )

  val lpath_shamt = exp - Mux1H(
    Seq(fp32toint32,
      fp16toint16 || fp16toint32
    ),
    Seq((VectorFloat.expBias(f32.expWidth) + f32.fracWidth).U,
      (VectorFloat.expBias(f16.expWidth) + f16.fracWidth).U)
  )
  val lpath_max_shamt = Mux1H(
    Seq(fp32toint32,
      fp16toint16,
      fp16toint32),
    Seq((31 - f32.fracWidth).U,
      (15 - f16.fracWidth).U,
      (31 - f16.fracWidth).U)
  )
  val lpath_sig_shifted = Wire(UInt(32.W))
  val lpath_sig_shifted_reg0 = RegNext(lpath_sig_shifted)
  lpath_sig_shifted := Mux1H(
    Seq(fp32toint32,
      fp16toint16 || fp16toint32),
    Seq((sig << lpath_shamt(lpath_max_shamt.getWidth - 1, 0))(31, 0),
      (sig << lpath_shamt(lpath_max_shamt.getWidth - 1, 0))(31, 13))
  )

  val lpath_iv = Wire(Bool())
  val lpath_iv_reg0 = RegNext(lpath_iv, false.B)

  lpath_iv := (fp32toint32 || fp16toint16 || fp16toint32) && !is_signed_int && sign

  val lpath_of = Wire(Bool())
  val lpath_of_reg0 = RegNext(lpath_of)
  lpath_of := (fp32toint32 || fp16toint16 || fp16toint32) && is_signed_int && (exp === max_int_exp) && (!sign || (sign && in.tail(9).orR))

  // right
  val rpath_shamt = Mux1H(
    Seq(fp32toint32,
      fp32toint16,
      fp16toint16 || fp16toint32,
      fp16toint8),
    Seq((VectorFloat.expBias(f32.expWidth) + f32.fracWidth).U,
      (VectorFloat.expBias(f32.expWidth) + 15).U,
      (VectorFloat.expBias(f16.expWidth) + f16.fracWidth).U,
      (VectorFloat.expBias(f16.expWidth) + 8 - 1).U)
  ) - exp

  val rpath_sig_shifted0 = Wire(UInt(25.W))
  val rpath_sig_shifted_reg0 = RegNext(rpath_sig_shifted0, 0.U(25.W))
  val (rpath_sig_shifted, rpath_sticky) = ShiftRightJam(Cat(sig, 0.U), rpath_shamt)
  rpath_sig_shifted0 := rpath_sig_shifted

  // int2fp
  val in_abs = Mux1H(
    Seq(is_int2fp && sign,
      is_int2fp && !sign),
    Seq((~in).asUInt + 1.U,
      in)
  )

  val int2fp_clz = Mux1H(
    Seq(out_is_fp32 || exp_of,
      int32tofp16 && !in_abs.tail(1).head(16).orR || int16tofp16 || int8tofp16,
    ),
    Seq(CLZ(in_abs(31, 0)),
      CLZ(in_abs(15, 0)))
  )

  val in_shift = Mux1H(
    Seq(out_is_fp32 || exp_of,
      int32tofp16 && !in_abs.tail(1).head(16).orR || int16tofp16 || int8tofp16
    ),
    Seq((in_abs.tail(1) << int2fp_clz)(30, 0),
      Cat((in_abs.tail(1) << int2fp_clz)(14, 0), 0.U(16.W)))
  )

  val exp_raw = Wire(UInt(8.W))
  val exp_raw_reg0 = RegNext(exp_raw, 0.U(8.W))
  exp_raw := Mux1H(
    Seq(is_int2fp && out_is_fp32,
      is_int2fp && out_is_fp16),
    Seq(VectorFloat.expBias(f32.expWidth).asUInt +& 31.U - int2fp_clz,
      VectorFloat.expBias(f16.expWidth).asUInt +& 15.U - int2fp_clz)
  )

  // share RoundingUnit
  round_in := Mux1H(
    Seq(fp32toint32,
      fp32toint16,
      fp16toint16 || fp16toint32,
      fp16toint8,
      int16tofp32,
      int32tofp32,
      int8tofp16,
      int16tofp16 || int32tofp16),
    Seq(rpath_sig_shifted.head(f32.precision),
      rpath_sig_shifted.head(16),
      rpath_sig_shifted.head(f16.precision),
      rpath_sig_shifted.head(8),
      in_shift.head(16),
      in_shift.head(23),
      in_shift.head(8),
      in_shift.head(10))
  )
  round_roundIn := Mux1H(
    Seq(fp32toint32,
      fp32toint16,
      fp16toint16 || fp16toint32,
      fp16toint8,
      int16tofp32,
      int32tofp32,
      int8tofp16,
      int16tofp16 || int32tofp16),
    Seq(rpath_sig_shifted.tail(f32.precision).head(1),
      rpath_sig_shifted.tail(16).head(1),
      rpath_sig_shifted.tail(f16.precision).head(1),
      rpath_sig_shifted.tail(8).head(1),
      in_shift.tail(16).head(1),
      in_shift.tail(23).head(1),
      in_shift.tail(8).head(1),
      in_shift.tail(10).head(1)
    )
  )
  round_stickyIn := Mux1H(
    Seq(fp32toint32,
      fp32toint16,
      fp16toint16 || fp16toint32,
      fp16toint8,
      int16tofp32,
      int32tofp32,
      int8tofp16,
      int16tofp16 || int32tofp16),
    Seq(rpath_sticky,
      rpath_sticky || rpath_sig_shifted.tail(17).orR,
      rpath_sticky || rpath_sig_shifted.tail(12).orR,
      rpath_sticky || rpath_sig_shifted.tail(9).orR,
      in_shift.tail(16).orR,
      in_shift.tail(f32.precision).orR,
      in_shift.tail(8).orR,
      in_shift.tail(f16.precision).orR
    )
  )
  round_signIn := (is_fp2int || is_int2fp) && sign

  val sel_lpath = Wire(Bool())
  val sel_lpath_reg0 = RegNext(sel_lpath)
  sel_lpath := exp >= Mux1H(
    Seq(is_fp2int && in_is_fp32,
      is_fp2int && in_is_fp16),
    Seq((VectorFloat.expBias(f32.expWidth) + f32.fracWidth).U,
      (VectorFloat.expBias(f16.expWidth) + f16.fracWidth).U)
  )

  val max_int = Mux1H(
    Seq(fp32toint32 || fp16toint32,
      fp32toint16 || fp16toint16,
      fp16toint8),
    Seq(Cat(!is_signed_int, ~0.U(31.W)),
      Cat(!is_signed_int, ~0.U(15.W)),
      Cat(!is_signed_int, ~0.U(7.W)))
  )
  val min_int = Mux1H(
    Seq(fp32toint32 || fp16toint32,
      fp32toint16 || fp16toint16,
      fp16toint8),
    Seq(Cat(is_signed_int, 0.U(31.W)),
      Cat(is_signed_int, 0.U(15.W)),
      Cat(is_signed_int, 0.U(7.W)))
  )
  val max_min_int = Wire(UInt(32.W))
  val max_min_int_reg0 = RegNext(max_min_int, 0.U(32.W))

  val sign_or_nan = Mux1H(
    Seq(fp32toint32 || fp32toint16,
      fp16toint32 || fp16toint16 || fp16toint8),
    Seq(fp_in.decode.isNaN,
      src.tail(4).head(f16.expWidth).andR && src.tail(9).orR)) | !sign

  max_min_int := Mux(sign_or_nan, max_int, min_int)

  // vfr
  is_normal := Mux1H(
    Seq(is_vfr && in_is_fp16,
      is_vfr && in_is_fp32),
    Seq(exp.tail(3).orR & !exp.tail(3).andR,
      exp.orR & !exp.andR)
  )

  val is_subnormal = Mux1H(
    Seq(is_vfr && in_is_fp16,
      is_vfr && in_is_fp32),
    Seq(!exp.tail(3).orR,
      !exp.orR)
  )
  is_inf := Mux1H(
    Seq(is_vfr && in_is_fp16,
      is_vfr && in_is_fp32),
    Seq(exp.tail(3).andR & !sig.tail(1).head(10).orR,
      exp.andR & !sig.tail(1).orR)
  )
  is_nan := Mux1H(
    Seq(is_vfr && in_is_fp16,
      is_vfr && in_is_fp32),
    Seq(exp.tail(3).andR & sig.tail(1).head(10).orR,
      exp.andR & sig.tail(1).orR)
  )
  is_neginf := is_vfr && sign & is_inf

  is_neginf_negzero := Mux1H(
    Seq(is_vfr && in_is_fp16,
      is_vfr && in_is_fp32),
    Seq(sign & (is_normal | is_subnormal & sig.tail(1).head(10).orR),
      sign & (is_normal | is_subnormal & sig.tail(1).orR))
  )

  is_negzero := Mux1H(
    Seq(is_vfr && in_is_fp16,
      is_vfr && in_is_fp32),
    Seq(sign & is_subnormal & !sig.tail(1).head(10).orR,
      sign & is_subnormal & !sig.tail(1).orR)
  )

  is_poszero := Mux1H(
    Seq(is_vfr && in_is_fp16,
      is_vfr && in_is_fp32),
    Seq(!sign & is_subnormal & !sig.tail(1).head(10).orR,
      !sign & is_subnormal & !sig.tail(1).orR)
  )

  val is_poszero_posinf = Mux1H(
    Seq(is_vfr && in_is_fp16,
      is_vfr && in_is_fp32),
    Seq(!sign & (is_normal | is_subnormal & sig.tail(1).head(10).orR),
      !sign & (is_normal | is_subnormal & sig.tail(1).orR))
  )
  is_snan := Mux1H(
    Seq(is_vfr,
      is_fp2fp && is_widen),
    Seq(!sig.tail(1).head(1).asBool && is_nan,
       fp_in.exp.tail(3).andR && fp_in.sig.head(10).orR && !fp_in.sig.head(1).asBool
    )
  )

  is_neg2_bplus1_b := Mux1H(
    Seq(is_vfr && in_is_fp16,
      is_vfr && in_is_fp32),
    Seq(sign & (exp.tail(3) === 30.U),
      sign & (exp === 254.U))
  )
  is_neg2_b_bminus1 := Mux1H(
    Seq(is_vfr && in_is_fp16,
      is_vfr && in_is_fp32),
    Seq(sign & (exp.tail(3) === 29.U),
      sign & (exp === 253.U))
  )
  is_neg2_negbminus1_negzero := Mux1H(
    Seq(is_vfr && in_is_fp16,
      is_vfr && in_is_fp32),
    Seq(sign & (sig.tail(1).head(2) === "b00".U) & is_subnormal & sig.tail(1).head(10).orR,
      sign & (sig.tail(1).head(2) === "b00".U) & is_subnormal & sig.tail(1).orR)
  )
  is_pos2_poszero_negbminus1 := Mux1H(
    Seq(is_vfr && in_is_fp16,
      is_vfr && in_is_fp32),
    Seq(!sign & (sig.tail(1).head(2) === "b00".U) & is_subnormal & sig.tail(1).head(10).orR,
      !sign & (sig.tail(1).head(2) === "b00".U) & is_subnormal & sig.tail(1).orR)
  )
  is_pos2_bminus1_b := Mux1H(
    Seq(is_vfr && in_is_fp16,
      is_vfr && in_is_fp32),
    Seq(!sign & (exp.tail(3) === 29.U),
      !sign & (exp === 253.U))
  )
  is_pos2_b_bplus1 := Mux1H(
    Seq(is_vfr && in_is_fp16,
      is_vfr && in_is_fp32),
    Seq(!sign & (exp.tail(3) === 30.U),
      !sign & (exp === 254.U))
  )

  val zero_minus_lzc_f16 = Mux(is_vfr && in_is_fp16, 0.U - CLZ(sig.tail(1).head(10)), 0.U)

  val zero_minus_lzc_f32 = Mux(is_vfr && in_is_fp32, 0.U - CLZ(sig.tail(1)), 0.U)

  val exp_normalized_5 = Wire(UInt(5.W))
  val exp_normalized_8 = Wire(UInt(8.W))
  val exp_normalized_5_reg0 = RegNext(exp_normalized_5)
  val exp_normalized_8_reg0 = RegNext(exp_normalized_8)
  exp_normalized_5 := Mux1H(
    Seq(in_is_fp16 && (is_vfrsqrt7 && is_poszero_posinf && is_normal || (is_vfrec7 && is_normal)),
       in_is_fp16 && (is_vfrsqrt7 && is_poszero_posinf && !is_normal || (is_vfrec7 && !is_normal))),
    Seq(exp(4,0),
      Cat(Fill(f16.expWidth - zero_minus_lzc_f16.getWidth, zero_minus_lzc_f16.head(1)), zero_minus_lzc_f16))
  )
  exp_normalized_8 := Mux1H(
    Seq(in_is_fp32 && is_vfrsqrt7 && is_poszero_posinf && is_normal || (is_vfrec7 && is_normal),
      in_is_fp32 && is_vfrsqrt7 && is_poszero_posinf && !is_normal || (is_vfrec7 && !is_normal)),
    Seq(exp,
      Cat(Fill(f32.expWidth - zero_minus_lzc_f32.getWidth, zero_minus_lzc_f32.head(1)), zero_minus_lzc_f32))
  )

  val sig_normalized_11 = Wire(UInt(11.W))
  val sig_normalized_11_reg0 = RegNext(sig_normalized_11)
  val sig_normalized_24 = Wire(UInt(24.W))
  val sig_normalized_24_reg0 = RegNext(sig_normalized_24)

  sig_normalized_11 := Mux1H(
    Seq(in_is_fp16 && (is_vfrsqrt7 && is_poszero_posinf && is_normal || (is_vfrec7 && is_normal)),
      in_is_fp16 && (is_vfrsqrt7 && is_poszero_posinf && !is_normal || (is_vfrec7 && !is_normal))),
    Seq(Cat(0.U, sig.tail(1).head(10)),
      (sig.tail(1).head(10) << 1.U).asUInt)
  )
  sig_normalized_24 := Mux1H(
    Seq(in_is_fp32 && (is_vfrsqrt7 && is_poszero_posinf && is_normal || (is_vfrec7 && is_normal)),
      in_is_fp32 && (is_vfrsqrt7 && is_poszero_posinf && !is_normal || (is_vfrec7 && !is_normal))),
    Seq(Cat(0.U, sig.tail(1)),
      (sig.tail(1) << 1.U).asUInt)
  )
  val clz_fp16 = Wire(UInt(4.W))
  val clz_fp32 = Wire(UInt(8.W))
  val clz_fp16_reg0 = RegNext(clz_fp16)
  val clz_fp32_reg0 = RegNext(clz_fp32)

  clz_fp16 := Mux(in_is_fp16 && is_vfr, CLZ(sig_normalized_11), 0.U)
  clz_fp32 := Mux(in_is_fp32 && is_vfr, CLZ(sig_normalized_24), 0.U)

  val out_exp_normalized_fp16 = Mux(is_vfrec7 && in_is_fp16, 29.U - exp_normalized_5, 0.U)

  val out_exp_normalized_fp32 = Mux(is_vfrec7 && in_is_fp32, 253.U - exp_normalized_8, 0.U)

  val out_exp_zero_negone = Wire(Bool())
  val out_exp_zero_negone_reg0 = RegNext(out_exp_zero_negone)
  out_exp_zero_negone := Mux1H(
    Seq(is_vfrec7 && in_is_fp16,
      is_vfrec7 && in_is_fp32),
    Seq(!out_exp_normalized_fp16.orR || out_exp_normalized_fp16.andR,
      !out_exp_normalized_fp32.orR || out_exp_normalized_fp32.andR)
  )

  val out_exp_fp16 = Wire(UInt(5.W))
  val out_exp_fp16_reg0 = RegNext(out_exp_fp16, 0.U(5.W))
  out_exp_fp16 := Mux1H(
    Seq(in_is_fp16 && is_vfrsqrt7 && is_normal,
      in_is_fp16 && is_vfrsqrt7 && !is_normal,
      in_is_fp16 && is_vfrec7 && out_exp_zero_negone,
      in_is_fp16 && is_vfrec7 && !out_exp_zero_negone),
    Seq((44.U - exp.tail(3)) >> 1,
      (44.U + CLZ(sig.tail(1).head(10))) >> 1,
      0.U,
      out_exp_normalized_fp16)
  )
  val out_exp_fp32 = Wire(UInt(8.W))
  val out_exp_fp32_reg0 = RegNext(out_exp_fp32, 0.U(8.W))
  out_exp_fp32 := Mux1H(
    Seq(in_is_fp32 && is_vfrsqrt7 && is_normal,
      in_is_fp32 && is_vfrsqrt7 && !is_normal,
      in_is_fp32 && is_vfrec7 && out_exp_zero_negone,
      in_is_fp32 && is_vfrec7 && !out_exp_zero_negone
    ),
    Seq((380.U - exp) >> 1,
      (380.U + CLZ(sig.tail(1))) >> 1,
      0.U,
      out_exp_normalized_fp32)
  )

  val out_sign = Wire(Bool())
  val out_sign_reg0 = RegNext(out_sign)
  out_sign := is_vfrsqrt7 && is_poszero_posinf && sign


  // fp2fp narrow
  val exp_delta = VectorFloat.expBias(f32.expWidth) - VectorFloat.expBias(f16.expWidth)

  val down_exp_s = fp_in.exp.zext - exp_delta.S
  val down_exp = Wire(UInt(8.W))
  val down_exp_reg0 = RegNext(down_exp)
  down_exp := down_exp_s.asUInt

  // normal
  nor_in := fp_in.sig.head(f16.precision - 1)
  nor_roundBit := fp_in.sig.tail(f16.precision - 1).head(1).asBool
  nor_stickyBit := fp_in.sig.tail(f16.precision).orR
  nor_signBit := fp_in.sign

  // subnormal
  val shamt = (exp_delta + 1).U(f32.expWidth) - fp_in.exp
  val (subnor_sig, shift_sticky) = ShiftRightJam(Cat(fp_in.decode.expNotZero, fp_in.sig.head(f16.precision)), shamt)
  subnor_in := subnor_sig.tail(1).head(f16.precision - 1)
  subnor_roundBit := subnor_sig(0)
  subnor_stickyBit := shift_sticky | nor_stickyBit
  subnor_signBit := fp_in.sign

  val may_be_subnor = Wire(Bool())
  val may_be_subnor_reg0 = RegNext(may_be_subnor)
  may_be_subnor := down_exp.asSInt < 1.S

  val rmin = is_fp2fp && is_narrow &&  (io.rm === RTZ || (io.rm === RDN & !fp_in.sign) || (io.rm === RUP && fp_in.sign))

  val nor_of_exp = Wire(UInt(5.W))
  val nor_of_exp_reg0 = RegNext(nor_of_exp)
  nor_of_exp := Mux(rmin | io.opType(0).asBool, // ROD
    VectorFloat.maxNormExp(f16.expWidth).U(f16.expWidth.W),
    (VectorFloat.maxNormExp(f16.expWidth) + 1).U(f16.expWidth.W))

  val nor_of_sig = Wire(UInt(10.W))
  val nor_of_sig_reg0 = RegNext(nor_of_sig)
  nor_of_sig := Mux(rmin | io.opType(0).asBool, ~0.U((f16.precision - 1).W), 0.U((f16.precision - 1).W))

  val special_case = Wire(Bool())
  val special_case_reg0 = RegNext(special_case)
  special_case := is_fp2fp && is_narrow && fp_in.decode.expIsOnes

  val fp2fp_iv = Wire(Bool())
  val fp2fp_iv_reg0 = RegNext(fp2fp_iv)
  val fp2fp_dz = Wire(Bool())
  val fp2fp_dz_reg0 = RegNext(fp2fp_dz)
  fp2fp_iv := is_fp2fp && is_narrow && fp_in.decode.isSNaN
  fp2fp_dz := is_fp2fp && is_narrow && false.B

  val in_isnan_neg = Wire(Bool())
  val in_isnan_neg_reg0 = RegNext(in_isnan_neg)
  val in_sigNotZero = Wire(Bool())
  val in_sigNotZero_reg0 = RegNext(in_sigNotZero)
  in_isnan_neg := is_fp2fp && is_narrow && !fp_in.decode.isNaN && fp_in.sign
  in_sigNotZero := is_fp2fp && is_narrow && fp_in.decode.sigNotZero

  // fp2fp widen
  val nor_exp = Wire(UInt(8.W))
  val nor_exp_reg0 = RegNext(nor_exp)
  nor_exp := Mux(is_fp2fp && is_widen, exp_delta.U(f32.expWidth.W) + fp_in.exp, 0.U)

  val nor_sig = Wire(UInt(10.W))
  val nor_sig_reg0 = RegNext(nor_sig)
  nor_sig := Mux(is_fp2fp && is_widen, fp_in.sig.head(10), 0.U)

  val subnor_exp = Wire(UInt(8.W))
  val subnor_exp_reg0 = RegNext(subnor_exp)
  val subnor_sig_w = Wire(UInt(10.W))
  val subnor_sig_w_reg0 = RegNext(subnor_sig_w)
  val subnor_shamt = CLZ(fp_in.sig.head(10))
  subnor_exp := exp_delta.U(f32.expWidth.W) - subnor_shamt
  subnor_sig_w := Cat((fp_in.sig.head(10) << subnor_shamt)(f16.precision - 3, 0), 0.U(1.W))


  val isNaN = Wire(Bool())
  val expIsOnes = Wire(Bool())
  val expIsZero = Wire(Bool())
  val isZero = Wire(Bool())
  val isSubnormal = Wire(Bool())
  val sigNotZero = Wire(Bool())
  val isNaN_reg0 = RegNext(isNaN)
  val expIsOnes_reg0 = RegNext(expIsOnes)
  val expIsZero_reg0 = RegNext(expIsZero)
  val isZero_reg0 = RegNext(isZero)
  val isSubnormal_reg0 = RegNext(isSubnormal)
  val sigNotZero_reg0 = RegNext(sigNotZero)

  isNaN := is_fp2fp && is_widen && fp_in.exp.tail(3).andR && fp_in.sig.head(10).orR
  expIsOnes := is_fp2fp && is_widen && fp_in.exp.tail(3).andR
  expIsZero := is_fp2fp && is_widen && !fp_in.exp.tail(3).orR
  isZero := is_fp2fp && is_widen && !fp_in.exp.tail(3).orR && !fp_in.sig.head(10).orR
  isSubnormal := is_fp2fp && is_widen && !fp_in.exp.tail(3).orR && fp_in.sig.head(10).orR
  sigNotZero := is_fp2fp && is_widen && fp_in.sig.head(10).orR

  // cycle1
  val rounder = Module(new RoundingUnit(f32.precision))
  rounder.io.in := round_in_reg0
  rounder.io.roundIn := round_roundIn_reg0
  rounder.io.stickyIn := round_stickyIn_reg0
  rounder.io.signIn := round_signIn_reg0
  rounder.io.rm := rm_reg0

  val out_reg0 = Mux(is_fp2int_reg0 || is_int2fp_reg0, Mux(rounder.io.r_up, rounder.io.in + 1.U, rounder.io.in), 0.U)

  val cout_reg0 = rounder.io.r_up && Mux1H(
    Seq(fp32toint32_reg0 || fp16toint32_reg0,
      fp32toint16_reg0 || fp16toint16_reg0,
      fp16toint8_reg0,
      int16tofp32_reg0,
      int32tofp32_reg0,
      int8tofp16_reg0,
      int16tofp16_reg0 || int32tofp16_reg0
    ),
    Seq(rounder.io.in.andR,
      rounder.io.in.tail(9).andR,
      rounder.io.in.tail(17).andR,
      rounder.io.in.tail(8).andR,
      rounder.io.in.tail(1).andR,
      rounder.io.in.tail(16).andR,
      rounder.io.in.tail(14).andR
    )
  )

  val exp_reg0 = Mux1H(
    Seq(is_int2fp_reg0 && in_orR_reg0,
      is_int2fp_reg0 && !in_orR_reg0),
    Seq(exp_raw_reg0 + cout_reg0,
      0.U)
  )
  val sig_reg0 = Mux1H(
    Seq(fp32toint32_reg0,
      fp32toint16_reg0,
      fp16toint16_reg0 || fp16toint32_reg0,
      fp16toint8_reg0,
      is_int2fp_reg0),
    Seq(Cat(0.U((32 - f32.precision - 1).W), cout_reg0, out_reg0),
      Cat(0.U(8.W), out_reg0),
      Cat(0.U((32 - f16.precision - 1).W), cout_reg0, out_reg0(10, 0)),
      Cat(0.U(21.W), out_reg0(10, 0)),
      out_reg0)
  )

  val rpath_ix_reg0 = Mux1H(
    Seq(is_fp2int_reg0 && !fp16toint8_reg0,
      is_fp2int_reg0 && fp16toint8_reg0),
    Seq(rounder.io.inexact,
      rounder.io.inexact || rpath_sig_shifted_reg0(7,0).orR)
  )
  val rpath_iv_reg0 = is_fp2int_reg0 && !is_signed_int_reg0 && in_reg0.head(1).asBool && sig_reg0.orR

  val rpath_pos_of_reg0 = is_fp2int_reg0 && !in_reg0.head(1).asBool &&
    Mux(is_signed_int_reg0,
      Mux1H(
        Seq(fp32toint16_reg0,
          fp16toint8_reg0),
        Seq((in_reg0.tail(1).head(8) === 142.U) || (in_reg0.tail(1).head(8) === 141.U) && cout_reg0,
          (in_reg0.tail(1).head(8) === 22.U) || (in_reg0.tail(1).head(8) === 21.U) && cout_reg0)
      ),
      Mux1H(
        Seq(fp32toint16_reg0,
          fp16toint8_reg0),
        Seq((in_reg0.tail(1).head(8) === 142.U) && cout_reg0,
          (in_reg0.tail(1).head(8) === 22.U) && cout_reg0)
      )
    )
  val rpath_neg_of_reg0 = is_fp2int_reg0 && in_reg0.head(1).asBool && Mux1H(
    Seq(fp32toint16_reg0,
      fp16toint8_reg0),
    Seq(in_reg0.tail(1).head(8) === 142.U && (rounder.io.in.tail(8).orR || rounder.io.r_up),
      in_reg0.tail(1).head(8) === 22.U && (rounder.io.in.tail(17).orR || rounder.io.r_up))
  )
  val rpath_of_reg0 = Mux1H(
    Seq(fp32toint16_reg0 || fp16toint8_reg0,
      fp32toint32_reg0 || fp16toint16_reg0 || fp16toint32_reg0),
    Seq(rpath_neg_of_reg0 || rpath_pos_of_reg0,
      cout_reg0)
  )



  val vfrsqrt7Table = Module(new Rsqrt7Table)
  val vfrec7Table = Module(new Rec7Table)

  val sig_in7 = Mux1H(
    Seq(in_is_fp16_reg0 && is_vfrsqrt7_reg0,
      in_is_fp16_reg0 && is_vfrec7_reg0,
      in_is_fp32_reg0 && is_vfrsqrt7_reg0,
      in_is_fp32_reg0 && is_vfrec7_reg0),
    Seq(Cat(exp_normalized_5_reg0(0), (sig_normalized_11_reg0 << Mux(is_normal_reg0, 0.U, clz_fp16_reg0))(9, 4)),
      (sig_normalized_11_reg0 << Mux(is_normal_reg0, 0.U, clz_fp16_reg0))(9, 3),
      Cat(exp_normalized_8_reg0(0), (sig_normalized_24_reg0 << Mux(is_normal_reg0, 0.U, clz_fp32_reg0))(22, 17)),
      (sig_normalized_24_reg0 << Mux(is_normal_reg0, 0.U, clz_fp32_reg0))(22, 16))
  )

  vfrsqrt7Table.src := sig_in7
  vfrec7Table.src := sig_in7

  val sig_out7_reg0 = Wire(UInt(7.W))
  sig_out7_reg0 := Mux1H(
    Seq(is_vfrsqrt7_reg0,
      is_vfrec7_reg0),
    Seq(vfrsqrt7Table.out,
      vfrec7Table.out)
  )

  val out_sig_10_reg0 = Mux1H(
    Seq(in_is_fp16_reg0 && is_vfrec7_reg0 && out_exp_zero_negone_reg0 && (is_neg2_bplus1_b_reg0 || is_pos2_b_bplus1_reg0),
      in_is_fp16_reg0 && is_vfrec7_reg0 && out_exp_zero_negone_reg0 && (is_neg2_b_bminus1_reg0 || is_pos2_bminus1_b_reg0),
      in_is_fp16_reg0 && is_vfrec7_reg0 && out_exp_zero_negone_reg0 && !(is_neg2_bplus1_b_reg0 || is_pos2_b_bplus1_reg0 || is_neg2_b_bminus1_reg0 || is_pos2_bminus1_b_reg0),
      in_is_fp16_reg0 && is_vfrec7_reg0 && !out_exp_zero_negone_reg0),
    Seq(Cat(0.U, 1.U, sig_out7_reg0, 0.U),
      Cat(1.U, sig_out7_reg0, 0.U(2.W)),
      Cat(1.U, sig_out7_reg0, 0.U(2.W)) >> 1,
      Cat(sig_out7_reg0, 0.U(3.W)))
  )
  val out_sig_23_reg0 = Mux1H(
    Seq(in_is_fp32_reg0 && is_vfrec7_reg0 && out_exp_zero_negone_reg0 && (is_neg2_bplus1_b_reg0 || is_pos2_b_bplus1_reg0),
      in_is_fp32_reg0 && is_vfrec7_reg0 && out_exp_zero_negone_reg0 && (is_neg2_b_bminus1_reg0 || is_pos2_bminus1_b_reg0),
      in_is_fp32_reg0 && is_vfrec7_reg0 && out_exp_zero_negone_reg0 && !(is_neg2_bplus1_b_reg0 || is_pos2_b_bplus1_reg0 || is_neg2_b_bminus1_reg0 || is_pos2_bminus1_b_reg0),
      in_is_fp32_reg0 && is_vfrec7_reg0 && !out_exp_zero_negone_reg0),
    Seq(Cat(0.U, 1.U, sig_out7_reg0, 0.U(14.W)),
      Cat(1.U, sig_out7_reg0, 0.U(15.W)),
      Cat(1.U, sig_out7_reg0, 0.U(15.W)) >> 1,
      Cat(sig_out7_reg0, 0.U(16.W)))
  )

  val fp_result_fp16 = Wire(UInt(16.W))
  fp_result_fp16 := Mux1H(
    Seq(in_is_fp16_reg0 && is_vfrsqrt7_reg0,
      in_is_fp16_reg0 && is_vfrec7_reg0),
    Seq(Cat(out_sign_reg0, out_exp_fp16_reg0, sig_out7_reg0, 0.U(3.W)),
      Cat(sign_reg0, out_exp_fp16_reg0, out_sig_10_reg0))
  )
  val fp_result_fp32 = Wire(UInt(32.W))
  fp_result_fp32 := Mux1H(
    Seq(in_is_fp32_reg0 && is_vfrsqrt7_reg0,
      in_is_fp32_reg0 && is_vfrec7_reg0),
    Seq(Cat(out_sign_reg0, out_exp_fp32_reg0, sig_out7_reg0, 0.U(16.W)),
      Cat(sign_reg0, out_exp_fp32_reg0, out_sig_23_reg0))
  )

  val result_nan_fp16 = Wire(UInt(16.W))
  val result_nan_fp32 = Wire(UInt(32.W))

  result_nan_fp16 := Cat(0.U(1.W), Fill(6, 1.U), 0.U(9.W))
  result_nan_fp32 := Cat(0.U, Fill(9, 1.U), 0.U(22.W))

  val result_inf_fp16 = Wire(UInt(15.W))
  val result_inf_fp32 = Wire(UInt(31.W))

  result_inf_fp16 := Cat(Fill(5, 1.U), 0.U(10.W))
  result_inf_fp32 := Cat(Fill(8, 1.U), Fill(23, 0.U))

  val result_greatest_fin_fp16 = Wire(UInt(15.W))
  val result_greatest_fin_fp32 = Wire(UInt(31.W))
  result_greatest_fin_fp16 := Cat(Fill(4, 1.U), 0.U, Fill(10, 1.U))
  result_greatest_fin_fp32 := Cat(Fill(7, 1.U), 0.U, Fill(23, 1.U))


  val of_reg0 = Mux1H(
    Seq(is_fp2int_reg0,
      is_int2fp_reg0 && out_is_fp32_reg0,
      is_int2fp_reg0 && int32tofp16_reg0,
      is_int2fp_reg0 && !out_is_fp32_reg0 && !int32tofp16_reg0),
    Seq(exp_of_reg0 || sel_lpath_reg0 && lpath_of_reg0 || !sel_lpath_reg0 && rpath_of_reg0,
      exp_reg0 === 255.U,
      exp_of_reg0 || exp_reg0 === 31.U,
      exp_reg0 === 31.U)
  )
  val iv_reg0 = is_fp2int_reg0 && (of_reg0 || sel_lpath_reg0 && lpath_iv_reg0 || !sel_lpath_reg0 && rpath_iv_reg0)

  val ix_reg0 = Mux1H(
    Seq(is_fp2int_reg0,
      is_int2fp_reg0 && (out_is_fp32_reg0 || int16tofp16_reg0 || int8tofp16_reg0),
      int32tofp16_reg0),
    Seq(!iv_reg0 && !sel_lpath_reg0 && rpath_ix_reg0,
      rounder.io.inexact,
      exp_of_reg0 || rounder.io.inexact)
  )

  val result_fp = Mux1H(
    Seq(is_int2fp_reg0),
    Seq(Cat(sign_reg0, Mux(RoundingUnit.is_rmin(rm_reg0, sign_reg0), Cat(Fill(4, 1.U), 0.U, Fill(10, 1.U)), Cat(Fill(5, 1.U), Fill(10, 0.U)))))
  )

  val int_abs_reg0 = Mux1H(
    Seq(is_fp2int_reg0 && sel_lpath_reg0,
      is_fp2int_reg0 && !sel_lpath_reg0),
    Seq(lpath_sig_shifted_reg0,
      sig_reg0)
  )
  val int_reg0 = Mux1H(
    Seq(fp32toint16_reg0 || fp16toint16_reg0,
      fp32toint32_reg0 || fp16toint32_reg0,
      fp16toint8_reg0),
    Seq(Mux(in_reg0.head(1).asBool && is_signed_int_reg0, -int_abs_reg0.tail(16), int_abs_reg0.tail(16)),
      Mux(in_reg0.head(1).asBool && is_signed_int_reg0, -int_abs_reg0, int_abs_reg0),
      Mux(in_reg0.head(1).asBool && is_signed_int_reg0, -int_abs_reg0.tail(24), int_abs_reg0.tail(24)))
  )


  // fp2fp
  val nor_rounder = Module(new RoundingUnit(f16.precision - 1))
  nor_rounder.io.in := nor_in_reg0
  nor_rounder.io.roundIn := nor_roundBit_reg0
  nor_rounder.io.stickyIn := nor_stickyBit_reg0
  nor_rounder.io.signIn := nor_signBit_reg0
  nor_rounder.io.rm := rm_reg0

  val fp2fp_n_out_reg0 = Mux(nor_rounder.io.r_up, nor_rounder.io.in + 1.U, nor_rounder.io.in)
  val fp2fp_n_cout_reg0 = nor_rounder.io.r_up && nor_rounder.io.in.andR

  val nor_sig_rounded_reg0 = fp2fp_n_out_reg0
  val nor_exp_rounded_reg0 = Mux(fp2fp_n_cout_reg0, down_exp_reg0.asSInt + 1.S, down_exp_reg0.asSInt)
  val nor_of_reg0 =
    Mux(fp2fp_n_cout_reg0,
      down_exp_reg0.asSInt > (VectorFloat.maxNormExp(f16.expWidth) - 1).S,
      down_exp_reg0.asSInt > VectorFloat.maxNormExp(f16.expWidth).S).asBool

  val exp_uf_reg0 = Mux(fp2fp_n_cout_reg0, down_exp_reg0.asSInt < 0.S, down_exp_reg0.asSInt < 1.S).asBool
  val nor_ix_reg0 = nor_of_reg0 | nor_rounder.io.inexact

  val subnor_rounder = Module(new RoundingUnit(f16.precision - 1))
  subnor_rounder.io.in := subnor_in_reg0
  subnor_rounder.io.roundIn := subnor_roundBit_reg0
  subnor_rounder.io.stickyIn := subnor_stickyBit_reg0
  subnor_rounder.io.signIn := subnor_signBit_reg0
  subnor_rounder.io.rm := rm_reg0

  val fp2fp_w_out_reg0 = Mux(subnor_rounder.io.r_up, subnor_rounder.io.in + 1.U, subnor_rounder.io.in)
  val fp2fp_w_cout_reg0 = subnor_rounder.io.r_up && subnor_rounder.io.in.andR

  val subnor_sig_rounded_reg0 = fp2fp_w_out_reg0
  val subnor_exp_rounded_reg0 = Mux(fp2fp_w_cout_reg0, 1.U, 0.U)
  val subnor_ix_reg0 = subnor_rounder.io.inexact

  val common_exp = Mux1H(
    Seq(
      !may_be_subnor_reg0 & nor_of_reg0,
      !may_be_subnor_reg0 & !nor_of_reg0,
      may_be_subnor_reg0
    ),
    Seq(
      nor_of_exp_reg0,
      nor_exp_rounded_reg0(f16.expWidth - 1, 0),
      subnor_exp_rounded_reg0
    )
  )

  val common_sig = Mux1H(
    Seq(
      !may_be_subnor_reg0 & nor_of_reg0,
      !may_be_subnor_reg0 & !nor_of_reg0,
      may_be_subnor_reg0
    ),
    Seq(
      nor_of_sig_reg0,
      nor_sig_rounded_reg0,
      subnor_sig_rounded_reg0
    )
  )

  val fp2fp_of_reg0 = is_fp2fp_reg0 && is_narrow_reg0 && !special_case_reg0 && nor_of_reg0
  val fp2fp_uf_reg0 = is_fp2fp_reg0 && is_narrow_reg0 && !special_case_reg0 && may_be_subnor_reg0 && subnor_ix_reg0 && exp_uf_reg0
  val fp2fp_ix_reg0 = is_fp2fp_reg0 && is_narrow_reg0 && !special_case_reg0 && (!may_be_subnor_reg0 && nor_ix_reg0 || may_be_subnor_reg0 && subnor_ix_reg0)


  val result1H = VecInit(Seq(is_fp2int_reg0,

    int32tofp32_reg0,
    int16tofp32_reg0,
    int32tofp16_reg0 && of_reg0,
    int8tofp16_reg0,
    int32tofp16_reg0 && !of_reg0 || int16tofp16_reg0,

    in_is_fp16_reg0 && (is_vfrsqrt7_reg0 && (is_nan_reg0 || is_neginf_negzero_reg0) || is_vfrec7_reg0 && is_nan_reg0),
    in_is_fp16_reg0 && is_vfrsqrt7_reg0 && is_inf_reg0,
    in_is_fp16_reg0 && is_vfrsqrt7_reg0 && (is_negzero_reg0 || is_poszero_reg0),
    in_is_fp16_reg0 && (is_vfrsqrt7_reg0 && !(is_nan_reg0 || is_neginf_negzero_reg0 || is_inf_reg0 || is_negzero_reg0 || is_poszero_reg0) ||
      is_vfrec7_reg0 && !(is_nan_reg0 || is_inf_reg0 || is_negzero_reg0 || is_poszero_reg0 || is_neg2_negbminus1_negzero_reg0 || is_pos2_poszero_negbminus1_reg0)),
    in_is_fp16_reg0 && is_vfrec7_reg0 && is_inf_reg0,
    in_is_fp16_reg0 && is_vfrec7_reg0 && (is_negzero_reg0 || is_poszero_reg0),
    in_is_fp16_reg0 && is_vfrec7_reg0 && is_neg2_negbminus1_negzero_reg0,
    in_is_fp16_reg0 && is_vfrec7_reg0 && is_pos2_poszero_negbminus1_reg0,

    in_is_fp32_reg0 && (is_vfrsqrt7_reg0 && (is_nan_reg0 || is_neginf_negzero_reg0) || is_vfrec7_reg0 && is_nan_reg0),
    in_is_fp32_reg0 && is_vfrsqrt7_reg0 && is_inf_reg0,
    in_is_fp32_reg0 && is_vfrsqrt7_reg0 && (is_negzero_reg0 || is_poszero_reg0),
    in_is_fp32_reg0 && (is_vfrsqrt7_reg0 && !(is_nan_reg0 || is_neginf_negzero_reg0 || is_inf_reg0 || is_negzero_reg0 || is_poszero_reg0) ||
      is_vfrec7_reg0 && !(is_nan_reg0 || is_inf_reg0 || is_negzero_reg0 || is_poszero_reg0 || is_neg2_negbminus1_negzero_reg0 || is_pos2_poszero_negbminus1_reg0)),
    in_is_fp32_reg0 && is_vfrec7_reg0 && is_inf_reg0,
    in_is_fp32_reg0 && is_vfrec7_reg0 && (is_negzero_reg0 || is_poszero_reg0),
    in_is_fp32_reg0 && is_vfrec7_reg0 && is_neg2_negbminus1_negzero_reg0,
    in_is_fp32_reg0 && is_vfrec7_reg0 && is_pos2_poszero_negbminus1_reg0,

    is_fp2fp_reg0 && is_narrow_reg0,
    is_fp2fp_reg0 && is_widen_reg0
  )).asUInt
  dontTouch(result1H)

  result0 := Mux1H(
    result1H,
    Seq(Mux(iv_reg0, max_min_int_reg0, int_reg0),

      Cat(sign_reg0, exp_reg0, sig_reg0(22,0)),
      Cat(sign_reg0, exp_reg0, sig_reg0(15,0), 0.U(7.W)),
      result_fp,
      Cat(sign_reg0, exp_reg0(4, 0), sig_reg0(7, 0), 0.U(2.W)),
      Cat(sign_reg0, exp_reg0(4, 0), sig_reg0(9, 0)),

      result_nan_fp16,
      Mux(is_neginf_reg0, result_nan_fp16, 0.U),
      Mux(is_negzero_reg0, Cat(1.U, result_inf_fp16), Cat(0.U, result_inf_fp16)),
      fp_result_fp16,
      Mux(is_neginf_reg0, Cat(1.U, 0.U(15.W)), 0.U),
      Mux(is_negzero_reg0, Cat(Fill(6, 1.U), 0.U(10.W)), Cat(0.U(1.W), Fill(5, 1.U), 0.U(10.W))),
      Mux(rm_reg0 === RUP || rm_reg0 === RTZ, Cat(1.U, result_greatest_fin_fp16), Cat(1.U, result_inf_fp16)),
      Mux(rm_reg0 === RDN || rm_reg0 === RTZ, Cat(0.U, result_greatest_fin_fp16), Cat(0.U, result_inf_fp16)),

      result_nan_fp32,
      Mux(is_neginf_reg0, result_nan_fp32, 0.U),
      Mux(is_negzero_reg0, Cat(1.U, result_inf_fp32), Cat(0.U, result_inf_fp32)),
      fp_result_fp32,
      Mux(is_neginf_reg0, Cat(1.U, 0.U(31.W)), 0.U),
      Mux(is_negzero_reg0, Cat(Fill(9, 1.U), 0.U(23.W)), Cat(0.U(1.W), Fill(8, 1.U), 0.U(23.W))),
      Mux(rm_reg0 === RUP || rm_reg0 === RTZ, Cat(1.U, result_greatest_fin_fp32), Cat(1.U, result_inf_fp32)),
      Mux(rm_reg0 === RDN || rm_reg0 === RTZ, Cat(0.U, result_greatest_fin_fp32), Cat(0.U, result_inf_fp32)),

      Cat(
        in_isnan_neg_reg0,
        Mux1H(
          Seq(special_case_reg0, !special_case_reg0),
          Seq(~0.U(f16.expWidth.W), common_exp)
        ),
        Mux1H(
          Seq(special_case_reg0, !special_case_reg0),
          Seq(
            Cat(in_sigNotZero_reg0, 0.U((f16.precision - 2).W)),
            common_sig
          ))),
      Cat(
        !isNaN_reg0 & sign_reg0,
        Mux1H(
          Seq(
            expIsOnes_reg0,
            isZero_reg0,
            isSubnormal_reg0,
            !expIsOnes_reg0 & !expIsZero_reg0
          ),
          Seq(
            ~0.U(f32.expWidth.W),
            0.U(f32.expWidth.W),
            subnor_exp_reg0,
            nor_exp_reg0
          )
        ),
        Mux1H(
          Seq(
            expIsOnes_reg0,
            expIsZero_reg0,
            !expIsOnes_reg0 & !expIsZero_reg0
          ),
          Seq(
            Cat(sigNotZero_reg0, 0.U((f32.precision - 2).W)),
            Cat(subnor_sig_w_reg0, 0.U((f32.precision - f16.precision).W)),
            Cat(nor_sig_reg0, 0.U((f32.precision - f16.precision).W))
          )
        )
      )
    )
  )
  val fflags1H = VecInit(Seq(is_fp2int_reg0,
      is_int2fp_reg0,

      is_vfrsqrt7_reg0 && (is_nan_reg0 || is_neginf_negzero_reg0),
      is_vfrsqrt7_reg0 && is_inf_reg0,
      is_vfrsqrt7_reg0 && (is_negzero_reg0 || is_poszero_reg0) || is_vfrec7_reg0 && (is_negzero_reg0 || is_poszero_reg0),
      is_vfrsqrt7_reg0 && !(is_nan_reg0 || is_neginf_negzero_reg0 || is_inf_reg0 || is_negzero_reg0 || is_poszero_reg0) ||
        is_vfrec7_reg0 && (!(is_nan_reg0 || is_negzero_reg0 || is_poszero_reg0 || is_neg2_negbminus1_negzero_reg0 || is_pos2_poszero_negbminus1_reg0) || is_inf_reg0),
      is_vfrec7_reg0 && is_nan_reg0,
      is_vfrec7_reg0 && (is_neg2_negbminus1_negzero_reg0 || is_pos2_poszero_negbminus1_reg0),

      is_fp2fp_reg0 && is_narrow_reg0,
      is_fp2fp_reg0 && is_widen_reg0)
  ).asUInt
  dontTouch(fflags1H)

  fflags0 := Mux1H(
    fflags1H,
    Seq(Cat(iv_reg0, false.B, false.B, false.B, ix_reg0),
      Cat(false.B, false.B, of_reg0, false.B, ix_reg0),

      Mux(is_snan_reg0 || is_neginf_negzero_reg0, "b10000".U, "b00000".U),
      Mux(is_neginf_reg0, "b10000".U, "b00000".U),
      "b01000".U,
      0.U,
      Mux(is_snan_reg0, "b10000".U, "b00000".U),
      "b00101".U,

      Cat(fp2fp_iv_reg0, fp2fp_dz_reg0, fp2fp_of_reg0, fp2fp_uf_reg0, fp2fp_ix_reg0),
      Cat(is_snan_reg0, 0.U(4.W)))
  )

  // cycle2
  result := result0_reg0
  fflags := fflags0_reg0

  io.result := result
  io.fflags := fflags



}
