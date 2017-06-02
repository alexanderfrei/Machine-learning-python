

# объявление множеств
s = {3, 2, 3, 5}  # equal set([3,2,3,5])
s1 = {0}
s2 = {1, 1, 2, 12, 2, 3, 5}

# операции с множествами
union = s.union(s2)
inter = s.intersection(s2)
diff = s.difference(s2)
s_diff = s.symmetric_difference(s2)
subset = s.issubset(s2)
superset = s.issuperset(s2)
disjoint = s1.isdisjoint(s2)

print(s, 'union', s2, ':', union)
print(s, '/', s2, ':', inter)
print(s, 'diff', s2, ':', diff)  # вычитание слева
print(s, 'sym_diff', s2, ':', s_diff)
print(s, 'is subset of', s2, ':', subset)
print(s, 'is superset of', s2, ':', superset)
print(s1, 'is disjoint with', s2, ':', disjoint)  # непересекающиеся множества

