import STAGE54Kinetic

set_option autoImplicit false

namespace IsingStage54Torus

open IsingStage54Kinetic

/-- Explicit periodic coordinates, so chords across a chosen seam are included. -/
abbrev Site (L : Nat) := Fin L × Fin L × Fin L

def forward {L : Nat} (h : 3 ≤ L) (x : Fin L) : Fin L :=
  if a : x.val+1 < L then ⟨x.val+1,a⟩ else ⟨0,by omega⟩

def backward {L : Nat} (_ : 3 ≤ L) (x : Fin L) : Fin L :=
  if x.val=0 then ⟨L-1,by have := x.isLt; omega⟩ else ⟨x.val-1,by have := x.isLt; omega⟩

private theorem fv {L : Nat} (h : 3 ≤ L) (x : Fin L) :
    (forward h x).val = if x.val+1<L then x.val+1 else 0 := by
  unfold forward
  split <;> simp_all

private theorem bv {L : Nat} (h : 3 ≤ L) (x : Fin L) :
    (backward h x).val = if x.val=0 then L-1 else x.val-1 := by
  unfold backward
  split <;> simp_all

private theorem dirs {L : Nat} (h : 3 ≤ L) (x : Fin L) :
    forward h x ≠ x ∧ backward h x ≠ x ∧ forward h x ≠ backward h x := by
  have hx := x.isLt
  simp only [ne_eq,Fin.ext_iff,fv,bv]
  split <;> split <;> omega

@[simp] theorem back_forward {L : Nat} (h : 3 ≤ L) (x : Fin L) :
    backward h (forward h x) = x := by
  apply Fin.ext
  have hx := x.isLt
  simp only [bv,fv]
  repeat' split
  all_goals omega

@[simp] theorem forward_back {L : Nat} (h : 3 ≤ L) (x : Fin L) :
    forward h (backward h x) = x := by
  apply Fin.ext
  have hx := x.isLt
  simp only [fv,bv]
  repeat' split
  all_goals omega

def adjacent {L : Nat} (h : 3 ≤ L) : Site L → List (Site L)
  | (x,y,z) => [(forward h x,y,z), (backward h x,y,z),
     (x,forward h y,z), (x,backward h y,z), (x,y,forward h z), (x,y,backward h z)]

def torus {L : Nat} (h : 3 ≤ L) : Graph6 (Site L) where
  near := adjacent h
  six := by rintro ⟨x,y,z⟩; rfl
  distinct := by
    rintro ⟨x,y,z⟩
    obtain ⟨a,b,c⟩ := dirs h x
    obtain ⟨d,e,f⟩ := dirs h y
    obtain ⟨_,_,j⟩ := dirs h z
    simp [adjacent,a,b,c,d,e,f,j]
  loopless := by
    rintro ⟨x,y,z⟩
    obtain ⟨a,b,_⟩ := dirs h x
    obtain ⟨d,e,_⟩ := dirs h y
    obtain ⟨g,i,_⟩ := dirs h z
    simp [adjacent,Ne.symm a,Ne.symm b,Ne.symm d,Ne.symm e,Ne.symm g,Ne.symm i]
  symmetric := by
    rintro ⟨x,y,z⟩ q hq
    simp only [adjacent,List.mem_cons,List.not_mem_nil,or_false] at hq
    rcases hq with rfl|rfl|rfl|rfl|rfl|rfl <;> simp [adjacent]

/-- On every cubic torus of width at least three, including every even torus used
in the path proof. The numerator uses one common denominator 4^(5*(n+1)). -/
theorem torus_chord_bound {L : Nat} (h : 3 ≤ L) (root : Site L) (n : Nat) :
    ((tails (torus h) (n+1) root []).map (walkWeight (torus h) root)).sum ≤
      6 * 5^5 * (5*4^5)^n :=
  full_bound (torus h) root n

private def box4 := torus (L:=4) (by decide)
private def origin4 : Site 4 := (0,0,0)

/-- A straight three-step torus path closes a geometric chord at its last vertex. -/
theorem wrap_contact :
    chordTotal box4 (1,0,0) [origin4] [(2,0,0),(3,0,0)] = 1 ∧
    productWeight box4 (1,0,0) [origin4] [(2,0,0),(3,0,0)] = 5*4^14 := by decide

set_option maxRecDepth 12000 in
set_option maxHeartbeats 6000000 in
theorem box4_three_steps :
    (tails box4 3 origin4 []).length = 150 ∧
    ((tails box4 3 origin4 []).map (walkWeight box4 origin4)).sum * 2 = 315*4^15 := by
  decide

#print axioms torus
#print axioms torus_chord_bound
#print axioms wrap_contact
#print axioms box4_three_steps

end IsingStage54Torus
