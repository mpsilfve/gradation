#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator, encoder_states
import onmt.translate.translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

import pickle

def translate(opt):
    if opt.perturb_states != '':
        onmt.translate.translator.perturb_states = [int(s) for s in opt.perturb_states.split(",")]
    else:
        onmt.translate.translator.perturb_states = []
    onmt.translate.translator.scaling_factor = opt.scaling_factor

    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    if (opt.repr_file != "" or opt.perturb_states != '') and opt.batch_size != 1:
        logger.info("WARNING! When using --repr_file or --perturb_states, you should set --batch_size=1.")

    translator = build_translator(opt, logger=logger, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    shard_pairs = zip(src_shards, tgt_shards)

    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug,
            align_debug=opt.align_debug
            )

    if opt.repr_file != "":
        pickle.dump(encoder_states,open(opt.repr_file, "wb"))

def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
