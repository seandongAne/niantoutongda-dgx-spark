from backend.tools.solver.layout_solver import (
    CandidateRegion,
    LayoutProblem,
    PlacementUnit,
    solve_layout,
)


def _regions():
    return [
        CandidateRegion("bedside_right", "surface", capacity_units=2, near_power=True),
        CandidateRegion("desk_top", "surface", capacity_units=3, near_power=True),
        CandidateRegion("closet_lower", "shelf", capacity_units=4, near_power=False),
    ]


def _units():
    return [
        PlacementUnit("g_bedside", size_units=2, requires_power=True,
                      allowed_support=frozenset({"surface"})),
        PlacementUnit("g_desk", size_units=2, requires_power=True,
                      allowed_support=frozenset({"surface"})),
        PlacementUnit("g_storage", size_units=3, requires_power=False,
                      allowed_support=frozenset({"shelf", "floor"})),
    ]


def _scores():
    return {
        ("g_bedside", "bedside_right"): 10,
        ("g_bedside", "desk_top"): 3,
        ("g_desk", "desk_top"): 10,
        ("g_desk", "bedside_right"): 2,
        ("g_storage", "closet_lower"): 8,
    }


def test_feasible_plan_matches_expected():
    result = solve_layout(LayoutProblem(units=_units(), regions=_regions(), scores=_scores()))
    assert result.status == "PLAN_READY"
    assert result.assignments == {
        "g_bedside": "bedside_right",
        "g_desk": "desk_top",
        "g_storage": "closet_lower",
    }
    # 首选被禁后的替代区域:床头组合应能挪到桌面(容量 3 够)
    assert result.alternatives["g_bedside"] == "desk_top"


def test_power_infeasible_yields_new_space_incompatible():
    regions = [CandidateRegion("closet_lower", "shelf", 4, near_power=False)]
    units = [PlacementUnit("g_charge", 1, requires_power=True,
                           allowed_support=frozenset({"surface", "shelf"}))]
    result = solve_layout(LayoutProblem(units=units, regions=regions))
    assert result.status == "NEW_SPACE_INCOMPATIBLE"
    assert any("无电源证据" in c for c in result.conflicts)


def test_capacity_joint_infeasible():
    regions = [CandidateRegion("small_surface", "surface", 2, near_power=True)]
    units = [
        PlacementUnit("g_a", 2, allowed_support=frozenset({"surface"})),
        PlacementUnit("g_b", 2, allowed_support=frozenset({"surface"})),
    ]
    result = solve_layout(LayoutProblem(units=units, regions=regions))
    assert result.status == "NEW_SPACE_INCOMPATIBLE"


def test_mutex_and_colocate():
    regions = [
        CandidateRegion("r1", "surface", 4, near_power=True),
        CandidateRegion("r2", "surface", 4, near_power=True),
    ]
    units = [
        PlacementUnit("g1", 1, allowed_support=frozenset({"surface"})),
        PlacementUnit("g2", 1, allowed_support=frozenset({"surface"})),
        PlacementUnit("g3", 1, allowed_support=frozenset({"surface"})),
    ]
    problem = LayoutProblem(
        units=units, regions=regions,
        scores={("g1", "r1"): 5, ("g2", "r1"): 5, ("g3", "r1"): 5},
        co_locate=[("g1", "g2")], mutex=[("g1", "g3")],
    )
    result = solve_layout(problem)
    assert result.status == "PLAN_READY"
    assert result.assignments["g1"] == result.assignments["g2"]
    assert result.assignments["g1"] != result.assignments["g3"]


def test_determinism():
    problem = LayoutProblem(units=_units(), regions=_regions(), scores=_scores())
    r1 = solve_layout(problem)
    r2 = solve_layout(problem)
    assert r1.assignments == r2.assignments
    assert r1.objective == r2.objective
