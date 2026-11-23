import STAGE54PrismData

set_option autoImplicit false

namespace IsingStage54Prism

def prices : Array Nat := #[
  0,4181010088,2796622230,1408633261,0,4181010088,2796622230,1696196670,1287882389,549724752,
  0,0,2768775703,2484813406,929924211,0,2768775703,2772376814,2428165575,1815642656]

def theta : Nat := 900000000
def rawGain : Nat := 8362020128
def netGain : Nat := 8137020128
def units : Nat := 10^12

def cell (a b : Nat) : Nat := (lowerCells[a]?.getD #[])[b]?.getD 0
def fee (i : Nat) : Nat := prices[i]?.getD 0
def slot (m : Nat) : Nat :=
  (List.range 4).countP (fun j => m.testBit j) + if m.testBit 4 then 5 else 0
def degree (m : Nat) : Nat := 1+(List.range 5).countP (fun j => m.testBit j)

def allowed (a b : Nat) (chord : Bool) : Bool :=
  let k := if chord then 3 else 2
  decide (k ≤ degree a ∧ k ≤ degree b)

def requested (a b : Nat) (chord : Bool) : Nat :=
  let q := if chord then 10 else 0
  fee (slot a+q) + fee (slot b+q)

def extra (chord : Bool) : Nat := if chord then theta else 0

def pairModes := (List.range 32).flatMap fun a =>
  (List.range 32).flatMap fun b => [false,true].filter (allowed a b)

set_option maxRecDepth 12000 in
theorem labelled_cells :
    lowerCells.size=32 ∧ (lowerCells.toList.all fun row => row.size==32)=true ∧
      pairModes.length=1637 := by decide

private def passes (a b : Nat) (chord : Bool) :=
  !allowed a b chord || decide (24 + requested a b chord ≤ cell a b + extra chord)

set_option maxRecDepth 12000 in
set_option maxHeartbeats 20000000 in
theorem all_edge_checks : ∀ a b : Fin 32, ∀ c : Bool, passes a.val b.val c = true := by decide

/-- An untrusted integer m can replace the printed lower bound only after comparing the
two. This implication checks all masks, the path/chord distinction, and the rounding slack. -/
theorem allocation_of_input (a b : Fin 32) (c : Bool) (m : Nat)
    (physicalLower : cell a.val b.val ≤ m) (legal : allowed a.val b.val c = true) :
    24 + requested a.val b.val c ≤ m + extra c := by
  have table := all_edge_checks a b c
  simp [passes,legal] at table
  omega

def incidenceFee (mask i j d : Nat) : Nat :=
  if !mask.testBit d then 0 else
    let transverse := (List.range 6).countP fun s => s/2 != d/2 && mask.testBit s
    let opp := if d%2 == 0 then d+1 else d-1
    let base := transverse + if mask.testBit opp then 5 else 0
    fee (base + if d==i || d==j then 0 else 10)

def legalStar (mask i j : Nat) : Bool :=
  decide (i<j) && mask.testBit i && mask.testBit j

def starIncome (mask i j : Nat) := ((List.range 6).map (incidenceFee mask i j)).sum

def stars := (List.range 64).flatMap fun m =>
  (List.range 6).flatMap fun i => (List.range 6).filter (legalStar m i)

set_option maxRecDepth 12000 in
theorem star_domain : stars.length=240 := by decide

set_option maxRecDepth 12000 in
set_option maxHeartbeats 20000000 in
theorem all_vertex_checks : ∀ m : Fin 64, ∀ i j : Fin 6,
    legalStar m.val i.val j.val=true → rawGain ≤ starIncome m.val i.val j.val := by decide

theorem tight_vertex : starIncome 63 0 1 = rawGain := by decide

theorem scalar_checks :
    4 * netGain + theta = 4 * rawGain ∧ 4*(5*20000)^4 < 5*94639^4 := by decide

def expNum := 6*units^3 + 6*netGain*units^2 + 3*netGain^2*units + netGain^3
def expDen := 6*units^3

/-- expNum/expDen is the cubic Taylor polynomial. An integer identity fixes the
endpoint (U+P)/(U-P), with U=94639/20000. -/
theorem extrapolated_endpoint_arithmetic :
    20000*expNum < 94639*expDen ∧
    1051047402734946162883735364514439 * (94639*expDen-20000*expNum) =
      681844442968178837116264635485561 * (94639*expDen+20000*expNum) := by decide

#print axioms labelled_cells
#print axioms all_edge_checks
#print axioms allocation_of_input
#print axioms star_domain
#print axioms all_vertex_checks
#print axioms tight_vertex
#print axioms scalar_checks
#print axioms extrapolated_endpoint_arithmetic

end IsingStage54Prism
