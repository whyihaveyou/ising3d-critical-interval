import Std

set_option autoImplicit false

namespace IsingStage54Kinetic

/-- A simple undirected graph with a duplicate-free list of six neighbours at each vertex. -/
structure Graph6 (V : Type) where
  near : V → List V
  six : ∀ v, (near v).length = 6
  distinct : ∀ v, (near v).Nodup
  loopless : ∀ v, v ∉ near v
  symmetric : ∀ {v z}, z ∈ near v → v ∈ near z

variable {V : Type} [DecidableEq V]

def seen (G : Graph6 V) (v : V) (past : List V) :=
  (G.near v).filter (fun z => decide (z ∈ v :: past))

def available (G : Graph6 V) (v : V) (past : List V) :=
  (G.near v).filter (fun z => decide (z ∉ v :: past))

/-- Previously visited neighbours of the new endpoint, minus its predecessor. -/
def contacts (G : Graph6 V) (v : V) (past : List V) :=
  (seen G v past).length - 1

def HasPrevious (G : Graph6 V) (v : V) (past : List V) : Prop :=
  ∃ s ∈ past, s ∈ G.near v

private theorem earlier_count (pre : V) (older ns : List V) (hn : pre ∉ older) :
    ns.countP (fun z => decide (z ∈ pre :: older)) =
      ns.count pre + ns.countP (fun z => decide (z ∈ older)) := by
  induction ns with
  | nil => simp
  | cons z zs ih =>
    rw [List.countP_cons, List.countP_cons, ih]
    by_cases hz : z = pre
    · subst z
      simp [hn,Nat.add_right_comm]
    · by_cases ho : z ∈ older <;> simp [hz,ho,Nat.add_assoc]

/-- On an actual self-avoiding history this counts exactly the older adjacent
vertices, each once. The immediate predecessor and the current endpoint are excluded. -/
theorem contacts_eq_older (G : Graph6 V) (v pre : V) (older : List V)
    (prior : pre ∈ G.near v) (fresh : pre ∉ older) :
    contacts G v (pre :: older) = ((G.near v).filter (fun z => decide (z ∈ older))).length := by
  have eraseLoop : (G.near v).countP (fun z => decide (z ∈ v :: pre :: older)) =
      (G.near v).countP (fun z => decide (z ∈ pre :: older)) := by
    apply List.countP_congr
    intro z hz
    have ne : z ≠ v := fun h => G.loopless v (h ▸ hz)
    simp [ne]
  simp only [contacts,seen,← List.countP_eq_length_filter,eraseLoop,earlier_count pre _ _ fresh,
    (G.distinct v).count,if_pos prior]
  omega

theorem available_contacts (G : Graph6 V) (v : V) (past : List V)
    (hp : HasPrevious G v past) :
    (available G v past).length + contacts G v past = 5 := by
  obtain ⟨s,hs,hn⟩ := hp
  have hs' : s ∈ seen G v past := by simp [seen,hn,hs]
  have hpos := List.length_pos_iff_exists_mem.mpr ⟨s,hs'⟩
  have partition := List.length_eq_countP_add_countP
    (fun z => decide (z ∈ v :: past)) (l := G.near v)
  have eq : 6 = (seen G v past).length + (available G v past).length := by
    simpa [G.six,seen,available,List.countP_eq_length_filter] using partition
  unfold contacts
  omega

private theorem next_previous (G : Graph6 V) (v : V) (past : List V) {z : V}
    (h : z ∈ available G v past) : HasPrevious G z (v :: past) :=
  ⟨v, List.mem_cons_self, G.symmetric (List.mem_filter.mp h).1⟩

/-- Vertex factor after clearing the common denominator 4^5. -/
def factor (a : Nat) : Nat := 5^a * 4^(5-a)
def branch : Nat := 5 * 4^5
def terminal : Nat := 5^5

theorem localFactor (a : Nat) (h : a ≤ 5) :
    (5-a) * factor a ≤ branch ∧ factor a ≤ terminal := by
  exact (by decide : ∀ j : Fin 6,
    (5-j.val) * factor j.val ≤ branch ∧ factor j.val ≤ terminal) ⟨a,by omega⟩

/-- The continued coordinates after v for each legal n-step path. -/
def tails (G : Graph6 V) : Nat → V → List V → List (List V)
  | 0, _, _ => [[]]
  | n+1, v, past => (available G v past).flatMap fun z =>
      (tails G n z (v :: past)).map (z :: ·)

def admissible (G : Graph6 V) : V → List V → List V → Prop
  | _, _, [] => True
  | v, past, z :: zs => z ∈ G.near v ∧ z ∉ v :: past ∧ admissible G z (v :: past) zs

theorem tails_complete (G : Graph6 V) (n : Nat) (v : V) (past xs : List V) :
    xs ∈ tails G n v past ↔ xs.length = n ∧ admissible G v past xs := by
  induction n generalizing v past xs with
  | zero => cases xs <;> simp [tails,admissible]
  | succ n ih =>
    cases xs with
    | nil => simp [tails]
    | cons z zs =>
      simp [tails,List.mem_flatMap,List.mem_map,List.cons.injEq,available,admissible,ih,
        and_assoc,and_comm,and_left_comm]

omit [DecidableEq V] in
private theorem prefix_unique (keys : List V) (rows : V → List (List V))
    (hk : keys.Nodup) (hr : ∀ z ∈ keys, (rows z).Nodup) :
    (keys.flatMap fun z => (rows z).map (z :: ·)).Nodup := by
  induction keys with
  | nil => simp
  | cons a rest ih =>
    obtain ⟨fresh, tail⟩ := List.nodup_cons.mp hk
    rw [List.flatMap_cons,List.nodup_append]
    refine ⟨List.pairwise_map.mpr ((hr a List.mem_cons_self).imp
      (fun ne eq => ne (List.cons.inj eq).2)), ih tail (fun z hz => hr z (by simp [hz])), ?_⟩
    intro x hx y hy he
    obtain ⟨u,_,rfl⟩ := List.mem_map.mp hx
    obtain ⟨b,hb,hby⟩ := List.mem_flatMap.mp hy
    obtain ⟨w,_,rfl⟩ := List.mem_map.mp hby
    exact fresh ((List.cons.inj he).1 ▸ hb)

theorem tails_nodup (G : Graph6 V) (n : Nat) (v : V) (past : List V) :
    (tails G n v past).Nodup := by
  induction n generalizing v past with
  | zero => simp [tails]
  | succ n ih =>
    exact prefix_unique _ _ ((G.distinct v).filter _) (fun z _ => ih z (v :: past))

/-- Includes the current vertex and the terminal vertex, charging a length-zero suffix once. -/
def productWeight (G : Graph6 V) : V → List V → List V → Nat
  | v, past, [] => factor (contacts G v past)
  | v, past, z :: zs => factor (contacts G v past) * productWeight G z (v :: past) zs

def chordTotal (G : Graph6 V) : V → List V → List V → Nat
  | v, past, [] => contacts G v past
  | v, past, z :: zs => contacts G v past + chordTotal G z (v :: past) zs

/-- Each non-root neighbour charge is counted at its later endpoint. -/
theorem product_formula (G : Graph6 V) (v : V) (past xs : List V)
    (hp : HasPrevious G v past) (legal : admissible G v past xs) :
    chordTotal G v past xs ≤ 5*(xs.length+1) ∧
    productWeight G v past xs =
      5^(chordTotal G v past xs)*4^(5*(xs.length+1)-chordTotal G v past xs) := by
  induction xs generalizing v past with
  | nil =>
    have hlocal := available_contacts G v past hp
    simp only [List.length_nil,Nat.zero_add,Nat.mul_one,chordTotal,productWeight,factor]
    exact ⟨by omega,True.intro⟩
  | cons z zs ih =>
    obtain ⟨hnear,hnew,hlegal⟩ := legal
    have hz : z ∈ available G v past := List.mem_filter.mpr ⟨hnear,decide_eq_true hnew⟩
    obtain ⟨tail_bound,tail_eq⟩ := ih z (v :: past) (next_previous G v past hz) hlegal
    have hlocal := available_contacts G v past hp
    simp only [chordTotal,productWeight,List.length_cons]
    constructor
    · omega
    · have index : 5*(zs.length+1+1)-(contacts G v past+chordTotal G z (v::past) zs) =
            (5-contacts G v past) + (5*(zs.length+1)-chordTotal G z (v::past) zs) := by omega
      rw [tail_eq,index,Nat.pow_add,Nat.pow_add]
      simp only [factor,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

omit [DecidableEq V] in
private theorem sum_map_left (a : Nat) (f : List V → Nat) (xs : List (List V)) :
    (xs.map (fun x => a*f x)).sum = a*(xs.map f).sum := by
  induction xs with
  | nil => simp
  | cons x xs ih => simp [ih,Nat.mul_add]

private theorem flat_map_sum {α β : Type} (keys : List α) (sets : α → List β) (f : β → Nat) :
    ((keys.flatMap sets).map f).sum = (keys.map (fun z => ((sets z).map f).sum)).sum := by
  induction keys with
  | nil => simp
  | cons k ks ih => simp [List.flatMap_cons,List.map_append,ih]

private theorem sum_cap {α : Type} (keys : List α) (f : α → Nat) (B : Nat)
    (h : ∀ z ∈ keys, f z ≤ B) : (keys.map f).sum ≤ keys.length*B := by
  induction keys with
  | nil => simp
  | cons k ks ih =>
    simpa [Nat.succ_mul,Nat.add_comm] using Nat.add_le_add (h k List.mem_cons_self)
      (ih (fun z hz => h z (by simp [hz])))

def mass (G : Graph6 V) (n : Nat) (v : V) (past : List V) :=
  ((tails G n v past).map (productWeight G v past)).sum

private theorem mass_step (G : Graph6 V) (n : Nat) (v : V) (past : List V) :
    mass G (n+1) v past =
      ((available G v past).map fun z =>
        factor (contacts G v past) * mass G n z (v :: past)).sum := by
  simp only [mass,tails,flat_map_sum,List.map_map,Function.comp_def,productWeight,sum_map_left]

theorem mass_bound (G : Graph6 V) (n : Nat) (v : V) (past : List V)
    (hp : HasPrevious G v past) : mass G n v past ≤ terminal * branch^n := by
  induction n generalizing v past with
  | zero =>
    have hlocal := available_contacts G v past hp
    have hcap := localFactor (contacts G v past) (by omega)
    simpa [mass,tails,productWeight] using hcap.2
  | succ n ih =>
    have hlocal := available_contacts G v past hp
    have hcap := localFactor (contacts G v past) (by omega)
    rw [mass_step]
    have hsum : _ ≤ (available G v past).length *
        (factor (contacts G v past) * (terminal * branch^n)) :=
      sum_cap _ _ _ (fun z hz => Nat.mul_le_mul_left _ (ih z (v :: past)
        (next_previous G v past hz)))
    have hrow : (available G v past).length * factor (contacts G v past) ≤ branch := by
      have h := hcap.1
      rw [show (available G v past).length = 5-contacts G v past by omega]
      exact h
    calc
      _ ≤ _ := hsum
      _ = ((available G v past).length * factor (contacts G v past)) *
          (terminal * branch^n) := by rw [Nat.mul_assoc]
      _ ≤ branch*(terminal*branch^n) := Nat.mul_le_mul_right _ hrow
      _ = _ := by simp [Nat.pow_succ,Nat.mul_left_comm,Nat.mul_comm]

def walkWeight (G : Graph6 V) (root : V) : List V → Nat
  | [] => 1
  | z :: zs => productWeight G z [root] zs

/-- Numerator of the full chord-weighted path sum with denominator 4^(5*(n+1)).
Every rooted path of length n+1 occurs exactly once by tails_complete and tails_nodup. -/
theorem full_bound (G : Graph6 V) (root : V) (n : Nat) :
    ((tails G (n+1) root []).map (walkWeight G root)).sum ≤ 6 * terminal * branch^n := by
  have h := sum_cap (available G root []) (fun z => mass G n z [root]) (terminal*branch^n)
    (fun z hz => mass_bound G n z [root] (next_previous G root [] hz))
  have len : (available G root []).length ≤ 6 := by
    exact Nat.le_trans (List.length_filter_le _ _) (Nat.le_of_eq (G.six root))
  simpa [tails,flat_map_sum,List.map_map,Function.comp_def,walkWeight,mass,Nat.mul_assoc]
    using Nat.le_trans h (Nat.mul_le_mul_right _ len)

#print axioms available_contacts
#print axioms contacts_eq_older
#print axioms localFactor
#print axioms tails_complete
#print axioms tails_nodup
#print axioms product_formula
#print axioms mass_bound
#print axioms full_bound

end IsingStage54Kinetic
