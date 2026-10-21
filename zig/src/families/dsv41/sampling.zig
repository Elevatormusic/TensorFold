//! The served target sampler of a decode or verify row (tensorfold.cuda.sampling.sample_rows with
//! exact_sampling.choose_rows): temperature 0 takes the row's argmax (the first of equal values). Otherwise, with top_k
//! > 0, the row's top_k + MARGIN largest logits (torch.topk's set; of equal values the lower ids) ordered by value, then
//! id (np.lexsort), the first top_k of them; each candidate's score is its logit over the temperature plus a Gumbel draw
//! keyed by (seed, position, token id) through a splitmix64 chain, so a verify row draws as the serial run at its
//! position. top_k 0 takes the nucleus rule instead (nucleusRow, cuda.sampling.nucleus_rows). top_p sets the scores
//! past the smallest prefix whose probability mass reaches top_p to -inf (the scaled
//! logits' exp over their sum: numpy's pairwise sum, then cumulative sums in order); min_p those under ln(min_p) of the
//! top. The token is the first best score. float64 as numpy computes it on the served image (numpy 2.1: its exp and log
//! are libm's, a row's sum of 20 its 8-accumulator pairwise sum; zrec np_libm.py).
const std = @import("std");

/// exact_sampling.Sampling (the served lane's defaults: temperature 1.0, top_k 20, top_p 0.95, min_p 0).
pub const Sampling = struct {
    seed: u64,
    temperature: f64 = 1.0,
    top_k: usize = 20,
    top_p: f64 = 0.95,
    min_p: f64 = 0.0,
};

/// exact_sampling.MARGIN: candidates beyond top_k read from the GPU.
pub const margin = 8;
/// The candidates a row takes at most here (top_k + MARGIN).
pub const max_candidates = 64;

extern "c" fn exp(x: f64) f64;
extern "c" fn log(x: f64) f64;

fn mix(x0: u64) u64 {
    var x = x0;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

/// exact_sampling.uniform: a (0, 1) double from a splitmix64 hash of (seed, position, token id).
pub fn uniform(seed: u64, position: u64, id: u64) f64 {
    var x = mix(seed +% 0x9E3779B97F4A7C15);
    x = mix(x ^ (position *% 0xD1B54A32D192ED03));
    x = mix(x ^ id);
    return @as(f64, @floatFromInt(x >> 11)) * 0x1p-53 + 0x1p-54;
}

/// numpy's pairwise sum of a contiguous float64 row (umath pairwise_sum: blocks of 128 summed 8 lanes wide, longer
/// rows split in halves at a multiple of 8).
pub fn pairwiseSum(a: []const f64) f64 {
    const n = a.len;
    if (n < 8) {
        var res: f64 = 0.0;
        for (a) |v| res += v;
        return res;
    }
    if (n > 128) {
        var n2 = n / 2;
        n2 -= n2 % 8;
        return pairwiseSum(a[0..n2]) + pairwiseSum(a[n2..]);
    }
    var r: [8]f64 = a[0..8].*;
    var i: usize = 8;
    while (i < n - n % 8) : (i += 8) {
        for (0..8) |j| r[j] += a[i + j];
    }
    var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    while (i < n) : (i += 1) res += a[i];
    return res;
}

/// a before b in np.lexsort((ids, -values))'s order: the larger value, then the lower id.
fn before(va: f32, ia: i64, vb: f32, ib: i64) bool {
    return va > vb or (va == vb and ia < ib);
}

/// The row's first `count` (value, id) in that order: torch.topk's set (of equal values the lower ids) as np.lexsort
/// orders it.
pub fn topCandidates(row: []const f32, count: usize, values: []f32, ids: []i64) usize {
    const k = @min(count, row.len);
    var n: usize = 0;
    for (row, 0..) |v, i| {
        const id: i64 = @intCast(i);
        if (n == k and !before(v, id, values[k - 1], ids[k - 1])) continue;
        // insert in order, dropping the last when full
        var j: usize = if (n < k) n else k - 1;
        while (j > 0 and before(v, id, values[j - 1], ids[j - 1])) : (j -= 1) {
            values[j] = values[j - 1];
            ids[j] = ids[j - 1];
        }
        values[j] = v;
        ids[j] = id;
        if (n < k) n += 1;
    }
    return n;
}

/// The first index of the largest value (np.argmax / torch.argmax on a row without NaN).
pub fn argmax(row: []const f32) usize {
    var best: usize = 0;
    for (row, 0..) |v, i| {
        if (v > row[best]) best = i;
    }
    return best;
}

/// sample_rows for one row of fp32 logits at absolute position `position`: the token.
pub fn sampleRow(row: []const f32, position: u64, s: Sampling) !i64 {
    if (s.temperature <= 0) return @intCast(argmax(row));
    if (s.top_k == 0) return nucleusRow(row, position, s);
    const count = @min(row.len, s.top_k + margin);
    if (count <= max_candidates) {
        var values: [max_candidates]f32 = undefined;
        var ids: [max_candidates]i64 = undefined;
        var scaled: [max_candidates]f64 = undefined;
        var score: [max_candidates]f64 = undefined;
        return choose(row, position, s, values[0..count], ids[0..count], scaled[0..count], score[0..count]);
    }
    // a top_k past max_candidates - margin: the same rule on heap buffers (no per-stream error that fails a round)
    const gpa = std.heap.page_allocator;
    const values = try gpa.alloc(f32, count);
    defer gpa.free(values);
    const ids = try gpa.alloc(i64, count);
    defer gpa.free(ids);
    const scaled = try gpa.alloc(f64, count);
    defer gpa.free(scaled);
    const score = try gpa.alloc(f64, count);
    defer gpa.free(score);
    return choose(row, position, s, values, ids, scaled, score);
}

/// choose_rows for one row on buffers of the candidate count: the top candidates in lexsort's order, the keyed scores,
/// the top_p and min_p cuts, the first best score.
fn choose(row: []const f32, position: u64, s: Sampling, values: []f32, ids: []i64, scaled: []f64, score: []f64) i64 {
    const width = topCandidates(row, values.len, values, ids);
    const k = @max(1, @min(s.top_k, width));
    const t = @max(s.temperature, 1e-6);
    for (0..k) |j| {
        scaled[j] = @as(f64, values[j]) / t;
        const u = uniform(s.seed, position, @bitCast(ids[j]));
        score[j] = scaled[j] - log(-log(u));
    }
    if (0.0 < s.top_p and s.top_p < 1.0) {
        var mx = scaled[0];
        for (scaled[1..k]) |v| mx = @max(mx, v);
        // probs over the candidates: the score buffer's twin (score is written after this only for the cut)
        var small: [max_candidates]f64 = undefined;
        const big = if (k > max_candidates) std.heap.page_allocator.alloc(f64, k) catch null else null;
        defer if (big) |b| std.heap.page_allocator.free(b);
        const probs: []f64 = if (big) |b| b else small[0..k];
        if (k > max_candidates and big == null) return ids[0]; // out of memory: the top candidate
        for (0..k) |j| probs[j] = exp(scaled[j] - mx);
        const sum = pairwiseSum(probs[0..k]);
        for (0..k) |j| probs[j] /= sum;
        var acc: f64 = 0.0;
        var below: usize = 0;
        for (0..k) |j| {
            acc += probs[j];
            if (acc < s.top_p) below += 1;
        }
        const keep = below + 1;
        for (keep..k) |j| score[j] = -std.math.inf(f64);
    }
    if (s.min_p > 0.0) {
        const floor = scaled[0] + log(s.min_p);
        for (0..k) |j| {
            if (scaled[j] < floor) score[j] = -std.math.inf(f64);
        }
    }
    var best: usize = 0;
    for (0..k) |j| {
        if (score[j] > score[best]) best = j;
    }
    return ids[best];
}

/// cuda.sampling.MASS: a token's share of the mass in fixed point (floor(exp(scaled - top) * 2^40)).
pub const mass_unit = 0x1p40;

/// ceil(Fraction(p) * total): the nucleus's mass need, exact (p's double as a rational, total < 2^58).
pub fn massNeed(p: f64, total: u64) u64 {
    const bits: u64 = @bitCast(p);
    const exp_bits: u64 = (bits >> 52) & 0x7ff;
    const frac: u64 = bits & ((@as(u64, 1) << 52) - 1);
    const m: u128 = if (exp_bits == 0) frac else frac | (@as(u64, 1) << 52);
    const e: i64 = if (exp_bits == 0) -1074 else @as(i64, @intCast(exp_bits)) - 1075;
    const prod: u128 = m * total;
    if (e >= 0) return @intCast(prod << @intCast(e));
    const shift: u64 = @intCast(-e);
    if (shift >= 120) return if (prod > 0) 1 else 0;
    const s: u7 = @intCast(shift);
    return @intCast((prod + ((@as(u128, 1) << s) - 1)) >> s);
}

const Candidate = struct { v: f64, id: u32, mass: u64 };

fn candidateBefore(_: void, a: Candidate, b: Candidate) bool {
    return a.v > b.v or (a.v == b.v and a.id < b.id);
}

/// nucleus_rows for one row (top_k off): the keyed draw over the top_p nucleus, then min_p. Each token's mass is
/// floor(exp(scaled - top) * 2^40); the nucleus is the shortest prefix of the (value, then id) order whose mass sum
/// reaches ceil(top_p * total); min_p keeps the prefix within ln(min_p) of the top; the token is the first best
/// scaled + Gumbel score of the kept prefix. Without a top_p cut and min_p, every token of the row is kept.
/// The whole row is on the host here, so the rule needs no candidate count (the Python one reads 1024 a rank first;
/// its result is the same). Mass uses libm's exp; the Python rule uses torch's CUDA exp, which can differ in the last
/// bit, so a token at a mass edge can differ from the Python engine; drafted and serial runs here use this one rule.
pub fn nucleusRow(row: []const f32, position: u64, s: Sampling) !i64 {
    const t = @max(s.temperature, 1e-6);
    var top: f64 = -std.math.inf(f64);
    for (row) |x| top = @max(top, @as(f64, x) / t);
    const cut = 0.0 < s.top_p and s.top_p < 1.0;
    const floor = if (s.min_p > 0.0) top + log(s.min_p) else -std.math.inf(f64);
    if (!cut) {
        // every token at or over the min_p floor (a prefix of the order); the first best score in that order
        var best: ?usize = null;
        var best_score: f64 = -std.math.inf(f64);
        var best_v: f64 = 0;
        for (row, 0..) |x, j| {
            const v = @as(f64, x) / t;
            if (v < floor) continue;
            const sc = v - log(-log(uniform(s.seed, position, j)));
            // equal scores: the earlier in (value, then id) order wins; ids rise with j
            if (best == null or sc > best_score or (sc == best_score and v > best_v)) {
                best = j;
                best_score = sc;
                best_v = v;
            }
        }
        return @intCast(best orelse argmax(row));
    }
    // the tokens with mass (a prefix of the order: mass falls with the value), sorted; the cut ends among them
    var total: u64 = 0;
    var n: usize = 0;
    for (row) |x| {
        const m: u64 = @intFromFloat(@floor(exp(@as(f64, x) / t - top) * mass_unit));
        total += m;
        if (m > 0) n += 1;
    }
    const cands = try std.heap.page_allocator.alloc(Candidate, n);
    defer std.heap.page_allocator.free(cands);
    var i: usize = 0;
    for (row, 0..) |x, j| {
        const v = @as(f64, x) / t;
        const m: u64 = @intFromFloat(@floor(exp(v - top) * mass_unit));
        if (m == 0) continue;
        cands[i] = .{ .v = v, .id = @intCast(j), .mass = m };
        i += 1;
    }
    std.mem.sort(Candidate, cands, {}, candidateBefore);
    const need = massNeed(s.top_p, total);
    var keep: usize = cands.len;
    var acc: u64 = 0;
    for (cands, 0..) |c, j| {
        acc += c.mass;
        if (acc >= need) {
            keep = j + 1;
            break;
        }
    }
    if (s.min_p > 0.0) {
        var within: usize = 0;
        for (cands[0..keep]) |c| {
            if (c.v >= floor) within += 1;
        }
        keep = @min(keep, within);
    }
    if (keep == 0) return @intCast(argmax(row));
    var best: usize = 0;
    var best_score: f64 = -std.math.inf(f64);
    for (cands[0..keep], 0..) |c, j| {
        const sc = c.v - log(-log(uniform(s.seed, position, c.id)));
        if (sc > best_score) {
            best = j;
            best_score = sc;
        }
    }
    return cands[best].id;
}

test "nucleus need is ceil(top_p * total) as a rational" {
    try std.testing.expectEqual(@as(u64, 95), massNeed(0.95, 100)); // 0.95's double is just under 0.95
    try std.testing.expectEqual(@as(u64, 50), massNeed(0.5, 100));
    try std.testing.expectEqual(@as(u64, 1), massNeed(0.5, 1));
    try std.testing.expectEqual(@as(u64, 1), massNeed(1e-300, 1 << 40));
}

test "nucleus: a peaked row keeps its top token, top_p 1 keeps every token" {
    var row: [64]f32 = undefined;
    for (&row, 0..) |*x, j| x.* = -@as(f32, @floatFromInt(j));
    row[7] = 40.0; // ~all of the mass
    for (0..50) |p| {
        try std.testing.expectEqual(@as(i64, 7), try nucleusRow(&row, p, .{ .seed = 3, .top_k = 0, .top_p = 0.9 }));
    }
    // a flat row with no cut: the draw spreads over the tokens and matches a direct keyed argmax
    const flat: [16]f32 = @splat(0);
    for (0..20) |p| {
        var best: u64 = 0;
        var bs: f64 = -std.math.inf(f64);
        for (0..16) |j| {
            const sc = -log(-log(uniform(5, p, j)));
            if (sc > bs) {
                bs = sc;
                best = j;
            }
        }
        try std.testing.expectEqual(@as(i64, @intCast(best)), try nucleusRow(&flat, p, .{ .seed = 5, .top_k = 0, .top_p = 1.0 }));
    }
}

test "nucleus: the cut, then min_p, as nucleus_rows draws them" {
    // scaled values 0, -ln 2, -ln 4, ...: masses 2^40, 2^39, 2^38, ...; top_p 0.7 needs 2 tokens (1 + 0.5 >= 0.7 * ~2)
    var row: [12]f32 = undefined;
    for (&row, 0..) |*x, j| x.* = -@as(f32, @floatFromInt(j)) * @as(f32, @floatCast(@log(@as(f64, 2.0))));
    for (0..200) |p| {
        const tok = try nucleusRow(&row, p, .{ .seed = 11, .top_k = 0, .top_p = 0.7 });
        try std.testing.expect(tok == 0 or tok == 1);
        const one = try nucleusRow(&row, p, .{ .seed = 11, .top_k = 0, .top_p = 0.7, .min_p = 0.75 });
        try std.testing.expectEqual(@as(i64, 0), one);
    }
}

test "splitmix64 draws as exact_sampling.uniform makes them" {
    // (seed, position, id) -> the bits of exact_sampling.uniform's double
    try std.testing.expectEqual(@as(u64, 0x3fe8cd8867d6a69c), @as(u64, @bitCast(uniform(101, 2229, 2581))));
    try std.testing.expectEqual(@as(u64, 0x3fd08f785f86da1b), @as(u64, @bitCast(uniform(404, 59, 129279))));
    try std.testing.expectEqual(@as(u64, 0x3fc9ff45ea7ce2be), @as(u64, @bitCast(uniform(0, 0, 0))));
}

test "numpy's pairwise sum" {
    const a = [_]f64{ 1e16, 1.0, -1e16, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 0.5, 0.25 };
    // r = a[0..8] (no whole second block of 8); ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)), then the rest in order
    var want = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
    for (a[8..]) |v| want += v;
    try std.testing.expectEqual(want, pairwiseSum(&a));
}

test "a top_k past the stack buffers draws as the same rule, and numpy's long pairwise sum" {
    var row: [300]f32 = undefined;
    for (&row, 0..) |*x, j| x.* = @as(f32, @floatFromInt(j % 37)) * 0.25;
    for (0..20) |p| {
        const t = try sampleRow(&row, p, .{ .seed = 9, .top_k = 100, .top_p = 0.9 });
        try std.testing.expect(t >= 0 and t < 300);
        try std.testing.expect(row[@intCast(t)] >= 4.0); // inside the nucleus of the larger values
    }
    var a: [300]f64 = undefined;
    for (&a, 0..) |*x, j| x.* = 1.0 / @as(f64, @floatFromInt(j + 1));
    // n 300: halves at 144 (150 - 150 % 8), then 144 and 156 (> 128: 72 + 84)
    const want = (pairwiseSum(a[0..72]) + pairwiseSum(a[72..144])) + (pairwiseSum(a[144..216]) + pairwiseSum(a[216..300]));
    try std.testing.expectEqual(want, pairwiseSum(&a));
}

test "top candidates by value, then id" {
    const row = [_]f32{ 1, 5, 3, 5, 2, 5, 0 };
    var v: [4]f32 = undefined;
    var ids: [4]i64 = undefined;
    try std.testing.expectEqual(@as(usize, 4), topCandidates(&row, 4, &v, &ids));
    try std.testing.expectEqualSlices(i64, &.{ 1, 3, 5, 2 }, &ids);
    try std.testing.expectEqualSlices(f32, &.{ 5, 5, 5, 3 }, &v);
}
