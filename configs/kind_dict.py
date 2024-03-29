kinds = {'UNEXPOSED_DECL': 0.0, 'STRUCT_DECL': 1.0, 'UNION_DECL': 2.0, 'CLASS_DECL': 3.0, 'ENUM_DECL': 4.0,
             'FIELD_DECL': 5.0, 'ENUM_CONSTANT_DECL': 6.0, 'FUNCTION_DECL': 7.0, 'VAR_DECL': 8.0, 'PARM_DECL': 9.0,
             'OBJC_INTERFACE_DECL': 10.0, 'OBJC_CATEGORY_DECL': 11.0, 'OBJC_PROTOCOL_DECL': 12.0,
             'OBJC_PROPERTY_DECL': 13.0, 'OBJC_IVAR_DECL': 14.0, 'OBJC_INSTANCE_METHOD_DECL': 15.0,
             'OBJC_CLASS_METHOD_DECL': 16.0, 'OBJC_IMPLEMENTATION_DECL': 17.0, 'OBJC_CATEGORY_IMPL_DECL': 18.0,
             'TYPEDEF_DECL': 19.0, 'CXX_METHOD': 20.0, 'NAMESPACE': 21.0, 'LINKAGE_SPEC': 22.0, 'CONSTRUCTOR': 23.0,
             'DESTRUCTOR': 24.0, 'CONVERSION_FUNCTION': 25.0, 'TEMPLATE_TYPE_PARAMETER': 26.0,
             'TEMPLATE_NON_TYPE_PARAMETER': 27.0, 'TEMPLATE_TEMPLATE_PARAMETER': 28.0, 'FUNCTION_TEMPLATE': 29.0,
             'CLASS_TEMPLATE': 30.0, 'CLASS_TEMPLATE_PARTIAL_SPECIALIZATION': 31.0, 'NAMESPACE_ALIAS': 32.0,
             'USING_DIRECTIVE': 33.0, 'USING_DECLARATION': 34.0, 'TYPE_ALIAS_DECL': 35.0, 'OBJC_SYNTHESIZE_DECL': 36.0,
             'OBJC_DYNAMIC_DECL': 37.0, 'CXX_ACCESS_SPEC_DECL': 38.0, 'OBJC_SUPER_CLASS_REF': 39.0,
             'OBJC_PROTOCOL_REF': 40.0, 'OBJC_CLASS_REF': 41.0, 'TYPE_REF': 42.0, 'CXX_BASE_SPECIFIER': 43.0,
             'TEMPLATE_REF': 44.0, 'NAMESPACE_REF': 45.0, 'MEMBER_REF': 46.0, 'LABEL_REF': 47.0,
             'OVERLOADED_DECL_REF': 48.0, 'VARIABLE_REF': 49.0, 'INVALID_FILE': 50.0, 'NO_DECL_FOUND': 51.0,
             'NOT_IMPLEMENTED': 52.0, 'INVALID_CODE': 53.0, 'UNEXPOSED_EXPR': 54.0, 'DECL_REF_EXPR': 55.0,
             'MEMBER_REF_EXPR': 56.0, 'CALL_EXPR': 57.0, 'OBJC_MESSAGE_EXPR': 58.0, 'BLOCK_EXPR': 59.0,
             'INTEGER_LITERAL': 60.0, 'FLOATING_LITERAL': 61.0, 'IMAGINARY_LITERAL': 62.0, 'STRING_LITERAL': 63.0,
             'CHARACTER_LITERAL': 64.0, 'PAREN_EXPR': 65.0, 'UNARY_OPERATOR': 66.0, 'ARRAY_SUBSCRIPT_EXPR': 67.0,
             'BINARY_OPERATOR': 68.0, 'COMPOUND_ASSIGNMENT_OPERATOR': 69.0, 'CONDITIONAL_OPERATOR': 70.0,
             'CSTYLE_CAST_EXPR': 71.0, 'COMPOUND_LITERAL_EXPR': 72.0, 'INIT_LIST_EXPR': 73.0, 'ADDR_LABEL_EXPR': 74.0,
             'StmtExpr': 75.0, 'GENERIC_SELECTION_EXPR': 76.0, 'GNU_NULL_EXPR': 77.0, 'CXX_STATIC_CAST_EXPR': 78.0,
             'CXX_DYNAMIC_CAST_EXPR': 79.0, 'CXX_REINTERPRET_CAST_EXPR': 80.0, 'CXX_CONST_CAST_EXPR': 81.0,
             'CXX_FUNCTIONAL_CAST_EXPR': 82.0, 'CXX_TYPEID_EXPR': 83.0, 'CXX_BOOL_LITERAL_EXPR': 84.0,
             'CXX_NULL_PTR_LITERAL_EXPR': 85.0, 'CXX_THIS_EXPR': 86.0, 'CXX_THROW_EXPR': 87.0, 'CXX_NEW_EXPR': 88.0,
             'CXX_DELETE_EXPR': 89.0, 'CXX_UNARY_EXPR': 90.0, 'OBJC_STRING_LITERAL': 91.0, 'OBJC_ENCODE_EXPR': 92.0,
             'OBJC_SELECTOR_EXPR': 93.0, 'OBJC_PROTOCOL_EXPR': 94.0, 'OBJC_BRIDGE_CAST_EXPR': 95.0,
             'PACK_EXPANSION_EXPR': 96.0, 'SIZE_OF_PACK_EXPR': 97.0, 'LAMBDA_EXPR': 98.0, 'OBJ_BOOL_LITERAL_EXPR': 99.0,
             'OBJ_SELF_EXPR': 100.0, 'OMP_ARRAY_SECTION_EXPR': 101.0, 'OBJC_AVAILABILITY_CHECK_EXPR': 102.0,
             'UNEXPOSED_STMT': 103.0, 'LABEL_STMT': 104.0, 'COMPOUND_STMT': 105.0, 'CASE_STMT': 106.0,
             'DEFAULT_STMT': 107.0, 'IF_STMT': 108.0, 'SWITCH_STMT': 109.0, 'WHILE_STMT': 110.0, 'DO_STMT': 111.0,
             'FOR_STMT': 112.0, 'GOTO_STMT': 113.0, 'INDIRECT_GOTO_STMT': 114.0, 'CONTINUE_STMT': 115.0,
             'BREAK_STMT': 116.0, 'RETURN_STMT': 117.0, 'ASM_STMT': 118.0, 'OBJC_AT_TRY_STMT': 119.0,
             'OBJC_AT_CATCH_STMT': 120.0, 'OBJC_AT_FINALLY_STMT': 121.0, 'OBJC_AT_THROW_STMT': 122.0,
             'OBJC_AT_SYNCHRONIZED_STMT': 123.0, 'OBJC_AUTORELEASE_POOL_STMT': 124.0, 'OBJC_FOR_COLLECTION_STMT': 125.0,
             'CXX_CATCH_STMT': 126.0, 'CXX_TRY_STMT': 127.0, 'CXX_FOR_RANGE_STMT': 128.0, 'SEH_TRY_STMT': 129.0,
             'SEH_EXCEPT_STMT': 130.0, 'SEH_FINALLY_STMT': 131.0, 'MS_ASM_STMT': 132.0, 'NULL_STMT': 133.0,
             'DECL_STMT': 134.0, 'OMP_PARALLEL_DIRECTIVE': 135.0, 'OMP_SIMD_DIRECTIVE': 136.0,
             'OMP_FOR_DIRECTIVE': 137.0, 'OMP_SECTIONS_DIRECTIVE': 138.0, 'OMP_SECTION_DIRECTIVE': 139.0,
             'OMP_SINGLE_DIRECTIVE': 140.0, 'OMP_PARALLEL_FOR_DIRECTIVE': 141.0,
             'OMP_PARALLEL_SECTIONS_DIRECTIVE': 142.0, 'OMP_TASK_DIRECTIVE': 143.0, 'OMP_MASTER_DIRECTIVE': 144.0,
             'OMP_CRITICAL_DIRECTIVE': 145.0, 'OMP_TASKYIELD_DIRECTIVE': 146.0, 'OMP_BARRIER_DIRECTIVE': 147.0,
             'OMP_TASKWAIT_DIRECTIVE': 148.0, 'OMP_FLUSH_DIRECTIVE': 149.0, 'SEH_LEAVE_STMT': 150.0,
             'OMP_ORDERED_DIRECTIVE': 151.0, 'OMP_ATOMIC_DIRECTIVE': 152.0, 'OMP_FOR_SIMD_DIRECTIVE': 153.0,
             'OMP_PARALLELFORSIMD_DIRECTIVE': 154.0, 'OMP_TARGET_DIRECTIVE': 155.0, 'OMP_TEAMS_DIRECTIVE': 156.0,
             'OMP_TASKGROUP_DIRECTIVE': 157.0, 'OMP_CANCELLATION_POINT_DIRECTIVE': 158.0, 'OMP_CANCEL_DIRECTIVE': 159.0,
             'OMP_TARGET_DATA_DIRECTIVE': 160.0, 'OMP_TASK_LOOP_DIRECTIVE': 161.0,
             'OMP_TASK_LOOP_SIMD_DIRECTIVE': 162.0, 'OMP_DISTRIBUTE_DIRECTIVE': 163.0,
             'OMP_TARGET_ENTER_DATA_DIRECTIVE': 164.0, 'OMP_TARGET_EXIT_DATA_DIRECTIVE': 165.0,
             'OMP_TARGET_PARALLEL_DIRECTIVE': 166.0, 'OMP_TARGET_PARALLELFOR_DIRECTIVE': 167.0,
             'OMP_TARGET_UPDATE_DIRECTIVE': 168.0, 'OMP_DISTRIBUTE_PARALLELFOR_DIRECTIVE': 169.0,
             'OMP_DISTRIBUTE_PARALLEL_FOR_SIMD_DIRECTIVE': 170.0, 'OMP_DISTRIBUTE_SIMD_DIRECTIVE': 171.0,
             'OMP_TARGET_PARALLEL_FOR_SIMD_DIRECTIVE': 172.0, 'OMP_TARGET_SIMD_DIRECTIVE': 173.0,
             'OMP_TEAMS_DISTRIBUTE_DIRECTIVE': 174.0, 'TRANSLATION_UNIT': 175.0, 'UNEXPOSED_ATTR': 176.0,
             'IB_ACTION_ATTR': 177.0, 'IB_OUTLET_ATTR': 178.0, 'IB_OUTLET_COLLECTION_ATTR': 179.0,
             'CXX_FINAL_ATTR': 180.0, 'CXX_OVERRIDE_ATTR': 181.0, 'ANNOTATE_ATTR': 182.0, 'ASM_LABEL_ATTR': 183.0,
             'PACKED_ATTR': 184.0, 'PURE_ATTR': 185.0, 'CONST_ATTR': 186.0, 'NODUPLICATE_ATTR': 187.0,
             'CUDACONSTANT_ATTR': 188.0, 'CUDADEVICE_ATTR': 189.0, 'CUDAGLOBAL_ATTR': 190.0, 'CUDAHOST_ATTR': 191.0,
             'CUDASHARED_ATTR': 192.0, 'VISIBILITY_ATTR': 193.0, 'DLLEXPORT_ATTR': 194.0, 'DLLIMPORT_ATTR': 195.0,
             'CONVERGENT_ATTR': 196.0, 'WARN_UNUSED_ATTR': 197.0, 'WARN_UNUSED_RESULT_ATTR': 198.0,
             'ALIGNED_ATTR': 199.0, 'PREPROCESSING_DIRECTIVE': 200.0, 'MACRO_DEFINITION': 201.0,
             'MACRO_INSTANTIATION': 202.0, 'INCLUSION_DIRECTIVE': 203.0, 'MODULE_IMPORT_DECL': 204.0,
             'TYPE_ALIAS_TEMPLATE_DECL': 205.0, 'STATIC_ASSERT': 206.0, 'FRIEND_DECL': 207.0,
             'OVERLOAD_CANDIDATE': 208.0}
