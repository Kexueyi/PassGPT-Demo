from Python3ParserVisitor import Python3ParserVisitor
from Python3Parser import Python3Parser


class MyPython3ParserVisitor(Python3ParserVisitor):

    def visitSingle_input(self, ctx: Python3Parser.Single_inputContext):
        print("visitSingle_input" + ': ' + ctx.getText())
        return super().visitSingle_input(ctx)

    def visitFile_input(self, ctx: Python3Parser.File_inputContext):
        print("visitFile_input" + ': ' + ctx.getText())
        return list(map(lambda c: self.visitStmt(c), ctx.stmt()))

    def visitEval_input(self, ctx: Python3Parser.Eval_inputContext):
        print("visitEval_input" + ': ' + ctx.getText())
        return super().visitEval_input(ctx)

    def visitDecorator(self, ctx: Python3Parser.DecoratorContext):
        print("visitDecorator" + ': ' + ctx.getText())
        return super().visitDecorator(ctx)

    def visitDecorators(self, ctx: Python3Parser.DecoratorsContext):
        print("visitDecorators" + ': ' + ctx.getText())
        return super().visitDecorators(ctx)

    def visitDecorated(self, ctx: Python3Parser.DecoratedContext):
        print("visitDecorated" + ': ' + ctx.getText())
        return super().visitDecorated(ctx)

    def visitAsync_funcdef(self, ctx: Python3Parser.Async_funcdefContext):
        fucDecl = self.visitFuncdef(ctx.funcdef())
        fucDecl.update([('async', True)])
        return fucDecl

    def visitFuncdef(self, ctx: Python3Parser.FuncdefContext):
        return {
            'type': 'Function declaration',
            'parameters': ctx.parameters().getText(),
            'body': ctx.getText(),
            'start': {'line': ctx.start.line, 'column': ctx.start.column},
            'end': {'line': ctx.stop.line, 'column': ctx.stop.column + len(ctx.stop.text)}
        }

    def visitParameters(self, ctx: Python3Parser.ParametersContext):
        print("visitParameters" + ': ' + ctx.getText())
        return super().visitParameters(ctx)

    def visitTypedargslist(self, ctx: Python3Parser.TypedargslistContext):
        print("visitTypedargslist" + ': ' + ctx.getText())
        return super().visitTypedargslist(ctx)

    def visitTfpdef(self, ctx: Python3Parser.TfpdefContext):
        print("visitTfpdef" + ': ' + ctx.getText())
        return super().visitTfpdef(ctx)

    def visitVarargslist(self, ctx: Python3Parser.VarargslistContext):
        print("visitVarargslist" + ': ' + ctx.getText())
        return super().visitVarargslist(ctx)

    def visitVfpdef(self, ctx: Python3Parser.VfpdefContext):
        print("visitVfpdef" + ': ' + ctx.getText())
        return super().visitVfpdef(ctx)

    def visitStmt(self, ctx: Python3Parser.StmtContext):
        print("visitStmt" + ': ' + ctx.getText())
        if ctx.simple_stmts():
            return self.visitSimple_stmts(ctx.simple_stmts())
        else:
            return self.visitCompound_stmt(ctx.compound_stmt())

    def visitSimple_stmts(self, ctx: Python3Parser.Simple_stmtsContext):
        print("visitSimple_stmts" + ': ' + ctx.getText())
        stmts = list(map(lambda c: self.visitSimple_stmt(c), ctx.simple_stmt()))
        return stmts if len(stmts) > 1 else stmts.pop()

    def visitSimple_stmt(self, ctx: Python3Parser.Simple_stmtContext):
        print("visitSimple_stmt" + ': ' + ctx.getText())
        if ctx.expr_stmt() is not None:
            return self.visitExpr_stmt(ctx.expr_stmt())
        elif ctx.del_stmt() is not None:
            return self.visitDel_stmt(ctx.del_stmt())
        elif ctx.pass_stmt() is not None:
            return self.visitPass_stmt(ctx.pass_stmt())
        elif ctx.flow_stmt() is not None:
            return self.visitFlow_stmt(ctx.flow_stmt())
        elif ctx.import_stmt() is not None:
            return self.visitImport_stmt(ctx.import_stmt())
        elif ctx.global_stmt() is not None:
            return self.visitGlobal_stmt(ctx.global_stmt())
        elif ctx.nonlocal_stmt() is not None:
            return self.visitNonlocal_stmt(ctx.nonlocal_stmt())
        elif ctx.assert_stmt() is not None:
            return self.visitAssert_stmt(ctx.assert_stmt())

    def visitExpr_stmt(self, ctx: Python3Parser.Expr_stmtContext):
        print("visitExpr_stmt" + ': ' + ctx.getText())
        if len(ctx.ASSIGN()) > 0:
            return {
                'type': 'Assignment',
                'body': ctx.getText(),
                'start': {'line': ctx.start.line, 'column': ctx.start.column},
                'end': {'line': ctx.stop.line, 'column': ctx.stop.column + len(ctx.stop.text)}
            }
        else:
            return self.visitChildren(ctx)

    def visitAnnassign(self, ctx: Python3Parser.AnnassignContext):
        print("visitAnnassign" + ': ' + ctx.getText())
        return super().visitAnnassign(ctx)

    def visitTestlist_star_expr(self, ctx: Python3Parser.Testlist_star_exprContext):
        print("visitTestlist_star_expr" + ': ' + ctx.getText())
        return super().visitTestlist_star_expr(ctx)

    def visitAugassign(self, ctx: Python3Parser.AugassignContext):
        print("visitAugassign" + ': ' + ctx.getText())
        return super().visitAugassign(ctx)

    def visitDel_stmt(self, ctx: Python3Parser.Del_stmtContext):
        print("visitDel_stmt" + ': ' + ctx.getText())
        return super().visitDel_stmt(ctx)

    def visitPass_stmt(self, ctx: Python3Parser.Pass_stmtContext):
        print("visitPass_stmt" + ': ' + ctx.getText())
        return super().visitPass_stmt(ctx)

    def visitFlow_stmt(self, ctx: Python3Parser.Flow_stmtContext):
        print("visitFlow_stmt" + ': ' + ctx.getText())
        return super().visitFlow_stmt(ctx)

    def visitBreak_stmt(self, ctx: Python3Parser.Break_stmtContext):
        print("visitBreak_stmt" + ': ' + ctx.getText())
        return super().visitBreak_stmt(ctx)

    def visitContinue_stmt(self, ctx: Python3Parser.Continue_stmtContext):
        print("visitContinue_stmt" + ': ' + ctx.getText())
        return super().visitContinue_stmt(ctx)

    def visitReturn_stmt(self, ctx: Python3Parser.Return_stmtContext):
        print("visitReturn_stmt" + ': ' + ctx.getText())
        return super().visitReturn_stmt(ctx)

    def visitYield_stmt(self, ctx: Python3Parser.Yield_stmtContext):
        print("visitYield_stmt" + ': ' + ctx.getText())
        return super().visitYield_stmt(ctx)

    def visitRaise_stmt(self, ctx: Python3Parser.Raise_stmtContext):
        print("visitRaise_stmt" + ': ' + ctx.getText())
        return super().visitRaise_stmt(ctx)

    def visitImport_stmt(self, ctx: Python3Parser.Import_stmtContext):
        print("visitImport_stmt" + ': ' + ctx.getText())
        return super().visitImport_stmt(ctx)

    def visitImport_name(self, ctx: Python3Parser.Import_nameContext):
        imports = []
        for names in self.visitDotted_as_names(ctx.dotted_as_names()):
            imp = {'type': 'Import',
                   'start': {'line': ctx.start.line, 'column': ctx.start.column},
                   'end': {'line': ctx.stop.line, 'column': ctx.stop.column + len(ctx.stop.text)}
                   }
            imp.update(names)
            imports.append(imp)
        return imports if len(imports) > 1 else imports.pop()

    def visitImport_from(self, ctx: Python3Parser.Import_fromContext):
        # return super().visitImport_from(ctx)
        return {
            'type': 'Import',
            'from': ctx.dotted_name().getText(),
            'import': '*' if ctx.STAR() is not None else self.visitImport_as_names(ctx.import_as_names()),
            'start': {'line': ctx.start.line, 'column': ctx.start.column},
            'end': {'line': ctx.stop.line, 'column': ctx.stop.column + len(ctx.stop.text)}
        }

    def visitImport_as_name(self, ctx: Python3Parser.Import_as_nameContext):
        return {
            'name': ctx.name(0).getText(),
            'alias': ctx.name(1).getText() if len(ctx.name()) > 1 else None
        }

    def visitDotted_as_name(self, ctx: Python3Parser.Dotted_as_nameContext):
        return {
            'name': ctx.dotted_name().getText(),
            'alias': ctx.name().getText() if ctx.AS() is not None else None
        }

    def visitImport_as_names(self, ctx: Python3Parser.Import_as_namesContext):
        lst = list(map(lambda c: self.visitImport_as_name(c), ctx.import_as_name()))
        return lst if len(lst) > 1 else lst.pop()

    def visitDotted_as_names(self, ctx: Python3Parser.Dotted_as_namesContext):
        return list(map(lambda c: self.visitDotted_as_name(c), ctx.dotted_as_name()))

    def visitDotted_name(self, ctx: Python3Parser.Dotted_nameContext):
        print("visitDotted_name" + ': ' + ctx.getText())
        return super().visitDotted_name(ctx)

    def visitGlobal_stmt(self, ctx: Python3Parser.Global_stmtContext):
        print("visitGlobal_stmt" + ': ' + ctx.getText())
        return super().visitGlobal_stmt(ctx)

    def visitNonlocal_stmt(self, ctx: Python3Parser.Nonlocal_stmtContext):
        print("visitNonlocal_stmt" + ': ' + ctx.getText())
        return super().visitNonlocal_stmt(ctx)

    def visitAssert_stmt(self, ctx: Python3Parser.Assert_stmtContext):
        print("visitAssert_stmt" + ': ' + ctx.getText())
        return super().visitAssert_stmt(ctx)

    def visitCompound_stmt(self, ctx: Python3Parser.Compound_stmtContext):
        print("visitCompound_stmt" + ': ' + ctx.getText())
        if ctx.if_stmt() is not None:
            return {
                'type': 'If Statement',
                'body': ctx.getText(),
                'start': {'line': ctx.start.line, 'column': ctx.start.column},
                'end': {'line': ctx.stop.line, 'column': ctx.stop.column + len(ctx.stop.text)}
            }
        elif ctx.while_stmt() is not None:
            return {
                'type': 'While Statement',
                'body': ctx.getText(),
                'start': {'line': ctx.start.line, 'column': ctx.start.column},
                'end': {'line': ctx.stop.line, 'column': ctx.stop.column + len(ctx.stop.text)}
            }
        elif ctx.for_stmt() is not None:
            return {
                'type': 'For Loop Statement',
                'body': ctx.getText(),
                'start': {'line': ctx.start.line, 'column': ctx.start.column},
                'end': {'line': ctx.stop.line, 'column': ctx.stop.column + len(ctx.stop.text)}
            }
        elif ctx.try_stmt() is not None:
            return {
                'type': 'Try Statement',
                'body': ctx.getText(),
                'start': {'line': ctx.start.line, 'column': ctx.start.column},
                'end': {'line': ctx.stop.line, 'column': ctx.stop.column + len(ctx.stop.text)}
            }
        elif ctx.with_stmt() is not None:
            return {
                'type': 'With Statement',
                'body': ctx.getText(),
                'start': {'line': ctx.start.line, 'column': ctx.start.column},
                'end': {'line': ctx.stop.line, 'column': ctx.stop.column + len(ctx.stop.text)}
            }
        elif ctx.funcdef() is not None:
            return {
                'type': 'Function declaration',
                'body': ctx.getText(),
                'start': {'line': ctx.start.line, 'column': ctx.start.column},
                'end': {'line': ctx.stop.line, 'column': ctx.stop.column + len(ctx.stop.text)}
            }
        elif ctx.classdef() is not None:
            return {
                'type': 'Class declaration',
                'body': ctx.getText(),
                'start': {'line': ctx.start.line, 'column': ctx.start.column},
                'end': {'line': ctx.stop.line, 'column': ctx.stop.column + len(ctx.stop.text)}
            }
        elif ctx.decorated() is not None:
            return {
                'type': 'Decorated declaration',
                'body': ctx.getText(),
                'start': {'line': ctx.start.line, 'column': ctx.start.column},
                'end': {'line': ctx.stop.line, 'column': ctx.stop.column + len(ctx.stop.text)}
            }
        elif ctx.async_stmt() is not None:
            return {
                'type': 'Async declaration',
                'body': ctx.getText(),
                'start': {'line': ctx.start.line, 'column': ctx.start.column},
                'end': {'line': ctx.stop.line, 'column': ctx.stop.column + len(ctx.stop.text)}
            }
        elif ctx.match_stmt() is not None:
            return {
                'type': 'Match declaration',
                'body': ctx.getText(),
                'start': {'line': ctx.start.line, 'column': ctx.start.column},
                'end': {'line': ctx.stop.line, 'column': ctx.stop.column + len(ctx.stop.text)}
            }

    def visitAsync_stmt(self, ctx: Python3Parser.Async_stmtContext):
        print("visitAsync_stmt" + ': ' + ctx.getText())
        return super().visitAsync_stmt(ctx)

    def visitIf_stmt(self, ctx: Python3Parser.If_stmtContext):
        print("visitIf_stmt" + ': ' + ctx.getText())
        return super().visitIf_stmt(ctx)

    def visitWhile_stmt(self, ctx: Python3Parser.While_stmtContext):
        print("visitWhile_stmt" + ': ' + ctx.getText())
        return super().visitWhile_stmt(ctx)

    def visitFor_stmt(self, ctx: Python3Parser.For_stmtContext):
        print("visitFor_stmt" + ': ' + ctx.getText())
        return super().visitFor_stmt(ctx)

    def visitTry_stmt(self, ctx: Python3Parser.Try_stmtContext):
        print("visitTry_stmt" + ': ' + ctx.getText())
        return super().visitTry_stmt(ctx)

    def visitWith_stmt(self, ctx: Python3Parser.With_stmtContext):
        print("visitWith_stmt" + ': ' + ctx.getText())
        return super().visitWith_stmt(ctx)

    def visitWith_item(self, ctx: Python3Parser.With_itemContext):
        print("visitWith_item" + ': ' + ctx.getText())
        return super().visitWith_item(ctx)

    def visitExcept_clause(self, ctx: Python3Parser.Except_clauseContext):
        print("visitExcept_clause" + ': ' + ctx.getText())
        return super().visitExcept_clause(ctx)

    def visitBlock(self, ctx: Python3Parser.BlockContext):
        print("visitBlock" + ': ' + ctx.getText())
        return self.visitChildren(ctx)

    def visitMatch_stmt(self, ctx: Python3Parser.Match_stmtContext):
        print("visitMatch_stmt" + ': ' + ctx.getText())
        return super().visitMatch_stmt(ctx)

    def visitSubject_expr(self, ctx: Python3Parser.Subject_exprContext):
        print("visitSubject_expr" + ': ' + ctx.getText())
        return super().visitSubject_expr(ctx)

    def visitStar_named_expressions(self, ctx: Python3Parser.Star_named_expressionsContext):
        print("visitStar_named_expressions" + ': ' + ctx.getText())
        return super().visitStar_named_expressions(ctx)

    def visitStar_named_expression(self, ctx: Python3Parser.Star_named_expressionContext):
        print("visitStar_named_expression" + ': ' + ctx.getText())
        return super().visitStar_named_expression(ctx)

    def visitCase_block(self, ctx: Python3Parser.Case_blockContext):
        print("visitCase_block" + ': ' + ctx.getText())
        return super().visitCase_block(ctx)

    def visitGuard(self, ctx: Python3Parser.GuardContext):
        print("visitGuard" + ': ' + ctx.getText())
        return super().visitGuard(ctx)

    def visitPatterns(self, ctx: Python3Parser.PatternsContext):
        print("visitPatterns" + ': ' + ctx.getText())
        return super().visitPatterns(ctx)

    def visitPattern(self, ctx: Python3Parser.PatternContext):
        print("visitPattern" + ': ' + ctx.getText())
        return super().visitPattern(ctx)

    def visitAs_pattern(self, ctx: Python3Parser.As_patternContext):
        print("visitAs_pattern" + ': ' + ctx.getText())
        return super().visitAs_pattern(ctx)

    def visitOr_pattern(self, ctx: Python3Parser.Or_patternContext):
        print("visitOr_pattern" + ': ' + ctx.getText())
        return super().visitOr_pattern(ctx)

    def visitClosed_pattern(self, ctx: Python3Parser.Closed_patternContext):
        print("visitClosed_pattern" + ': ' + ctx.getText())
        return super().visitClosed_pattern(ctx)

    def visitLiteral_pattern(self, ctx: Python3Parser.Literal_patternContext):
        print("visitLiteral_pattern" + ': ' + ctx.getText())
        return super().visitLiteral_pattern(ctx)

    def visitLiteral_expr(self, ctx: Python3Parser.Literal_exprContext):
        print("visitLiteral_expr" + ': ' + ctx.getText())
        return super().visitLiteral_expr(ctx)

    def visitComplex_number(self, ctx: Python3Parser.Complex_numberContext):
        print("visitComplex_number" + ': ' + ctx.getText())
        return super().visitComplex_number(ctx)

    def visitSigned_number(self, ctx: Python3Parser.Signed_numberContext):
        print("visitSigned_number" + ': ' + ctx.getText())
        return super().visitSigned_number(ctx)

    def visitSigned_real_number(self, ctx: Python3Parser.Signed_real_numberContext):
        print("visitSigned_real_number" + ': ' + ctx.getText())
        return super().visitSigned_real_number(ctx)

    def visitReal_number(self, ctx: Python3Parser.Real_numberContext):
        print("visitReal_number" + ': ' + ctx.getText())
        return super().visitReal_number(ctx)

    def visitImaginary_number(self, ctx: Python3Parser.Imaginary_numberContext):
        print("visitImaginary_number" + ': ' + ctx.getText())
        return super().visitImaginary_number(ctx)

    def visitCapture_pattern(self, ctx: Python3Parser.Capture_patternContext):
        print("visitCapture_pattern" + ': ' + ctx.getText())
        return super().visitCapture_pattern(ctx)

    def visitPattern_capture_target(self, ctx: Python3Parser.Pattern_capture_targetContext):
        print("visitPattern_capture_target" + ': ' + ctx.getText())
        return super().visitPattern_capture_target(ctx)

    def visitWildcard_pattern(self, ctx: Python3Parser.Wildcard_patternContext):
        print("visitWildcard_pattern" + ': ' + ctx.getText())
        return super().visitWildcard_pattern(ctx)

    def visitValue_pattern(self, ctx: Python3Parser.Value_patternContext):
        print("visitValue_pattern" + ': ' + ctx.getText())
        return super().visitValue_pattern(ctx)

    def visitAttr(self, ctx: Python3Parser.AttrContext):
        print("visitAttr" + ': ' + ctx.getText())
        return super().visitAttr(ctx)

    def visitName_or_attr(self, ctx: Python3Parser.Name_or_attrContext):
        print("visitName_or_attr" + ': ' + ctx.getText())
        return super().visitName_or_attr(ctx)

    def visitGroup_pattern(self, ctx: Python3Parser.Group_patternContext):
        print("visitGroup_pattern" + ': ' + ctx.getText())
        return super().visitGroup_pattern(ctx)

    def visitSequence_pattern(self, ctx: Python3Parser.Sequence_patternContext):
        print("visitSequence_pattern" + ': ' + ctx.getText())
        return super().visitSequence_pattern(ctx)

    def visitOpen_sequence_pattern(self, ctx: Python3Parser.Open_sequence_patternContext):
        print("visitOpen_sequence_pattern" + ': ' + ctx.getText())
        return super().visitOpen_sequence_pattern(ctx)

    def visitMaybe_sequence_pattern(self, ctx: Python3Parser.Maybe_sequence_patternContext):
        print("visitMaybe_sequence_pattern" + ': ' + ctx.getText())
        return super().visitMaybe_sequence_pattern(ctx)

    def visitMaybe_star_pattern(self, ctx: Python3Parser.Maybe_star_patternContext):
        print("visitMaybe_star_pattern" + ': ' + ctx.getText())
        return super().visitMaybe_star_pattern(ctx)

    def visitStar_pattern(self, ctx: Python3Parser.Star_patternContext):
        print("visitStar_pattern" + ': ' + ctx.getText())
        return super().visitStar_pattern(ctx)

    def visitMapping_pattern(self, ctx: Python3Parser.Mapping_patternContext):
        print("visitMapping_pattern" + ': ' + ctx.getText())
        return super().visitMapping_pattern(ctx)

    def visitItems_pattern(self, ctx: Python3Parser.Items_patternContext):
        print("visitItems_pattern" + ': ' + ctx.getText())
        return super().visitItems_pattern(ctx)

    def visitKey_value_pattern(self, ctx: Python3Parser.Key_value_patternContext):
        print("visitKey_value_pattern" + ': ' + ctx.getText())
        return super().visitKey_value_pattern(ctx)

    def visitDouble_star_pattern(self, ctx: Python3Parser.Double_star_patternContext):
        print("visitDouble_star_pattern" + ': ' + ctx.getText())
        return super().visitDouble_star_pattern(ctx)

    def visitClass_pattern(self, ctx: Python3Parser.Class_patternContext):
        print("visitClass_pattern" + ': ' + ctx.getText())
        return super().visitClass_pattern(ctx)

    def visitPositional_patterns(self, ctx: Python3Parser.Positional_patternsContext):
        print("visitPositional_patterns" + ': ' + ctx.getText())
        return super().visitPositional_patterns(ctx)

    def visitKeyword_patterns(self, ctx: Python3Parser.Keyword_patternsContext):
        print("visitKeyword_patterns" + ': ' + ctx.getText())
        return super().visitKeyword_patterns(ctx)

    def visitKeyword_pattern(self, ctx: Python3Parser.Keyword_patternContext):
        print("visitKeyword_pattern" + ': ' + ctx.getText())
        return super().visitKeyword_pattern(ctx)

    def visitTest(self, ctx: Python3Parser.TestContext):
        print("visitTest" + ': ' + ctx.getText())
        return super().visitTest(ctx)

    def visitTest_nocond(self, ctx: Python3Parser.Test_nocondContext):
        print("visitTest_nocond" + ': ' + ctx.getText())
        return super().visitTest_nocond(ctx)

    def visitLambdef(self, ctx: Python3Parser.LambdefContext):
        print("visitLambdef" + ': ' + ctx.getText())
        return super().visitLambdef(ctx)

    def visitLambdef_nocond(self, ctx: Python3Parser.Lambdef_nocondContext):
        print("visitLambdef_nocond" + ': ' + ctx.getText())
        return super().visitLambdef_nocond(ctx)

    def visitOr_test(self, ctx: Python3Parser.Or_testContext):
        print("visitOr_test" + ': ' + ctx.getText())
        return super().visitOr_test(ctx)

    def visitAnd_test(self, ctx: Python3Parser.And_testContext):
        print("visitAnd_test" + ': ' + ctx.getText())
        return super().visitAnd_test(ctx)

    def visitNot_test(self, ctx: Python3Parser.Not_testContext):
        print("visitNot_test" + ': ' + ctx.getText())
        return super().visitNot_test(ctx)

    def visitComparison(self, ctx: Python3Parser.ComparisonContext):
        print("visitComparison" + ': ' + ctx.getText())
        return super().visitComparison(ctx)

    def visitComp_op(self, ctx: Python3Parser.Comp_opContext):
        print("visitComp_op" + ': ' + ctx.getText())
        return super().visitComp_op(ctx)

    def visitStar_expr(self, ctx: Python3Parser.Star_exprContext):
        print("visitStar_expr" + ': ' + ctx.getText())
        return super().visitStar_expr(ctx)

    def visitExpr(self, ctx: Python3Parser.ExprContext):
        print("visitExpr" + ': ' + ctx.getText())
        return super().visitExpr(ctx)

    def visitAtom_expr(self, ctx: Python3Parser.Atom_exprContext):
        print("visitAtom_expr" + ': ' + ctx.getText())
        if (len(ctx.trailer()) > 0):
            if (ctx.trailer()[-1].OPEN_PAREN() is not None):
                names = list(map(lambda c: c.getText(), ctx.trailer()[0:len(ctx.trailer()) - 1]))
                names.insert(0, ctx.atom().getText())
                return {
                    'type': 'Function call',
                    'name': ''.join(names),
                    'arguments': ctx.trailer()[-1].getText(),
                    'start': {'line': ctx.start.line, 'column': ctx.start.column},
                    'end': {'line': ctx.stop.line, 'column': ctx.stop.column + len(ctx.stop.text)}
                }
        else:
            return self.visitChildren(ctx)

    def visitAtom(self, ctx: Python3Parser.AtomContext):
        return super().visitAtom(ctx)

    def visitName(self, ctx: Python3Parser.NameContext):
        print("visitName" + ': ' + ctx.getText())
        return super().visitName(ctx)

    def visitTestlist_comp(self, ctx: Python3Parser.Testlist_compContext):
        print("visitTestlist_comp" + ': ' + ctx.getText())
        return super().visitTestlist_comp(ctx)

    def visitTrailer(self, ctx: Python3Parser.TrailerContext):
        print("visitTrailer" + ': ' + ctx.getText())
        return super().visitTrailer(ctx)

    def visitSubscriptlist(self, ctx: Python3Parser.SubscriptlistContext):
        print("visitSubscriptlist" + ': ' + ctx.getText())
        return super().visitSubscriptlist(ctx)

    def visitSubscript_(self, ctx: Python3Parser.Subscript_Context):
        print("visitSubscript_" + ': ' + ctx.getText())
        return super().visitSubscript_(ctx)

    def visitSliceop(self, ctx: Python3Parser.SliceopContext):
        print("visitSliceop" + ': ' + ctx.getText())
        return super().visitSliceop(ctx)

    def visitExprlist(self, ctx: Python3Parser.ExprlistContext):
        print("visitExprlist" + ': ' + ctx.getText())
        return super().visitExprlist(ctx)

    def visitTestlist(self, ctx: Python3Parser.TestlistContext):
        print("visitTestlist" + ': ' + ctx.getText())
        return super().visitTestlist(ctx)

    def visitDictorsetmaker(self, ctx: Python3Parser.DictorsetmakerContext):
        print("visitDictorsetmaker" + ': ' + ctx.getText())
        return super().visitDictorsetmaker(ctx)

    def visitClassdef(self, ctx: Python3Parser.ClassdefContext):
        print("visitClassdef" + ': ' + ctx.getText())
        return super().visitClassdef(ctx)

    def visitArglist(self, ctx: Python3Parser.ArglistContext):
        print("visitArglist" + ': ' + ctx.getText())
        return super().visitArglist(ctx)

    def visitArgument(self, ctx: Python3Parser.ArgumentContext):
        print("visitArgument" + ': ' + ctx.getText())
        return super().visitArgument(ctx)

    def visitComp_iter(self, ctx: Python3Parser.Comp_iterContext):
        print("visitComp_iter" + ': ' + ctx.getText())
        return super().visitComp_iter(ctx)

    def visitComp_for(self, ctx: Python3Parser.Comp_forContext):
        print("visitComp_for" + ': ' + ctx.getText())
        return super().visitComp_for(ctx)

    def visitComp_if(self, ctx: Python3Parser.Comp_ifContext):
        print("visitComp_if" + ': ' + ctx.getText())
        return super().visitComp_if(ctx)

    def visitEncoding_decl(self, ctx: Python3Parser.Encoding_declContext):
        print("visitEncoding_decl" + ': ' + ctx.getText())
        return super().visitEncoding_decl(ctx)

    def visitYield_expr(self, ctx: Python3Parser.Yield_exprContext):
        print("visitYield_expr" + ': ' + ctx.getText())
        return super().visitYield_expr(ctx)

    def visitYield_arg(self, ctx: Python3Parser.Yield_argContext):
        print("visitYield_arg" + ': ' + ctx.getText())
        return super().visitYield_arg(ctx)

    def visitStrings(self, ctx: Python3Parser.StringsContext):
        print("visitStrings" + ': ' + ctx.getText())
        return super().visitStrings(ctx)


del Python3Parser
