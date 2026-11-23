import Std

set_option autoImplicit false

/-!
Stage 57: a reusable finite-certificate interface.

The intended use is deliberately modest.  An external enumerator emits a list
of natural-number rows and rational endpoints.  This file defines the
mathematical meaning of that output, a Boolean checker, and soundness theorems
for the checker.  It does not define a Gibbs measure, an infinite-volume
limit, or the critical point.  Those analytic bridges remain hypotheses of the
paper's main theorem.
-/

namespace IsingStage57

structure RationalEndpoint where
  num : Nat
  den : Nat
  deriving DecidableEq, Repr

def positiveDen (q : RationalEndpoint) : Prop := 0 < q.den

def crossLT (a b : RationalEndpoint) : Prop := a.num * b.den < b.num * a.den

def intervalOK (lo hi : RationalEndpoint) : Prop :=
  positiveDen lo ∧ positiveDen hi ∧ crossLT lo hi

def intervalCheck (lo hi : RationalEndpoint) : Bool :=
  decide (0 < lo.den ∧ 0 < hi.den ∧
    lo.num * hi.den < hi.num * lo.den)

theorem intervalCheck_sound {lo hi : RationalEndpoint}
    (h : intervalCheck lo hi = true) : intervalOK lo hi := by
  have h' := of_decide_eq_true h
  simpa [intervalOK, positiveDen, crossLT] using h'

theorem crossLT_irrelevant_to_float {lo hi : RationalEndpoint}
    (h : intervalOK lo hi) : lo.num * hi.den < hi.num * lo.den :=
  h.2.2

/- A row is the smallest useful finite arithmetic unit in the current
   certificates.  `combined` is the sum that the external program claims.
   The checker gives this claim a first-order mathematical meaning. -/
structure CertificateRow where
  ordinary : Nat
  collar : Nat
  combined : Nat
  deriving DecidableEq, Repr

def rowSemantics (r : CertificateRow) : Prop :=
  r.ordinary + r.collar = r.combined

def rowCheck (r : CertificateRow) : Bool :=
  decide (r.ordinary + r.collar = r.combined)

theorem rowCheck_sound {r : CertificateRow} (h : rowCheck r = true) :
    rowSemantics r := by
  have h' := of_decide_eq_true h
  simpa [rowCheck, rowSemantics] using h'

def rowsCheck : List CertificateRow → Bool
  | [] => true
  | r :: rs => rowCheck r && rowsCheck rs

theorem rowsCheck_sound (rs : List CertificateRow)
    (h : rowsCheck rs = true) :
    ∀ r, r ∈ rs → rowSemantics r := by
  induction rs with
  | nil =>
      intro r hr
      simp at hr
  | cons r rs ih =>
      cases hrow : rowCheck r with
      | false =>
          simp [rowsCheck, hrow] at h
      | true =>
          have hrest : rowsCheck rs = true := by
            simpa [rowsCheck, hrow] using h
          have hhead : rowSemantics r := rowCheck_sound hrow
          intro x hx
          simp only [List.mem_cons] at hx
          cases hx with
          | inl hxhead =>
              cases hxhead
              exact hhead
          | inr hxtail =>
              exact ih hrest x hxtail

/- The optional total field is useful when a checker must prove that all
   emitted rows add to a published total.  It is a finite theorem about the
   list, independent of any physical interpretation of the rows. -/
structure TotalCertificate where
  rows : List Nat
  total : Nat
  deriving DecidableEq, Repr

def totalSemantics (c : TotalCertificate) : Prop := c.rows.sum = c.total

def totalCheck (c : TotalCertificate) : Bool := decide (c.rows.sum = c.total)

theorem totalCheck_sound {c : TotalCertificate} (h : totalCheck c = true) :
    totalSemantics c := by
  have h' := of_decide_eq_true h
  simpa [totalCheck, totalSemantics] using h'

structure FiniteCertificate where
  lower : RationalEndpoint
  upper : RationalEndpoint
  rows : List CertificateRow
  rowTotal : TotalCertificate
  deriving DecidableEq, Repr

def certificateSemantics (c : FiniteCertificate) : Prop :=
  intervalOK c.lower c.upper ∧
  (∀ r, r ∈ c.rows → rowSemantics r) ∧
  totalSemantics c.rowTotal

def certificateCheck (c : FiniteCertificate) : Bool :=
  intervalCheck c.lower c.upper &&
    rowsCheck c.rows && totalCheck c.rowTotal

theorem certificateCheck_sound {c : FiniteCertificate}
    (h : certificateCheck c = true) : certificateSemantics c := by
  have htop : (intervalCheck c.lower c.upper = true ∧
      rowsCheck c.rows = true) ∧ totalCheck c.rowTotal = true := by
    simpa [certificateCheck] using h
  refine ⟨intervalCheck_sound htop.1.1, rowsCheck_sound c.rows htop.1.2,
    totalCheck_sound htop.2⟩

/- The exact endpoints used by the current CMP draft.  They are copied as
   integers, so this instance does not silently replace the published bound
   by a decimal approximation. -/
def stage57Lower : RationalEndpoint :=
  { num := 275442362401406827838944754146320499627
    den := 178669509598593172161055245853679500373 }

def stage57Upper : RationalEndpoint :=
  { num := 18874999999514449
    den := 11930000000000000 }

def stage57Rows : List CertificateRow :=
  [ { ordinary := 18928787824, collar := 277709212536,
      combined := 296638000360 },
    { ordinary := 8412, collar := 0, combined := 8412 } ]

/- This is the finite field-count metadata attached to the ordered-side
   certificate in the CMP supplement.  It is a cardinality identity only. -/
def stage57SourceClasses : Nat := 14
def stage57FieldsPerClass : Nat := 262144
def stage57CheckedFields : Nat := 3670016

theorem stage57_field_count_check :
    stage57SourceClasses * stage57FieldsPerClass = stage57CheckedFields := by
  decide

def stage57RowTotal : TotalCertificate :=
  { rows := [18928787824, 277709212536]
    total := 296638000360 }

def stage57Certificate : FiniteCertificate :=
  { lower := stage57Lower
    upper := stage57Upper
    rows := stage57Rows
    rowTotal := stage57RowTotal }

theorem stage57_interval_check :
    intervalCheck stage57Lower stage57Upper = true := by
  decide

theorem stage57_interval_sound :
    intervalOK stage57Lower stage57Upper := by
  exact intervalCheck_sound stage57_interval_check

theorem stage57_lower_lt_upper :
    stage57Lower.num * stage57Upper.den <
      stage57Upper.num * stage57Lower.den := by
  exact stage57_interval_sound.2.2

theorem stage57_rows_check : rowsCheck stage57Rows = true := by
  decide

theorem stage57_total_check : totalCheck stage57RowTotal = true := by
  decide

theorem stage57_certificate_check : certificateCheck stage57Certificate = true := by
  decide

theorem stage57_certificate_sound :
    certificateSemantics stage57Certificate := by
  exact certificateCheck_sound stage57_certificate_check

/- A deliberately transparent projection for paper supplements.  This is a
   finite arithmetic fact, not a theorem that the endpoints bound the physical
   critical coupling. -/
theorem stage57_published_integer_interface :
    positiveDen stage57Lower ∧ positiveDen stage57Upper ∧
      crossLT stage57Lower stage57Upper ∧
      (∀ r, r ∈ stage57Rows → rowSemantics r) ∧
      totalSemantics stage57RowTotal ∧
      stage57SourceClasses * stage57FieldsPerClass = stage57CheckedFields := by
  have h := stage57_certificate_sound
  rcases h with ⟨hi, hrs, ht⟩
  exact ⟨hi.1, hi.2.1, hi.2.2, hrs, ht, stage57_field_count_check⟩

#print axioms intervalCheck_sound
#print axioms rowsCheck_sound
#print axioms totalCheck_sound
#print axioms certificateCheck_sound
#print axioms stage57_certificate_sound
#print axioms stage57_published_integer_interface

end IsingStage57

