#!/usr/bin/perl
#
# Copyright 2017   Ewald Enzinger
# Apache 2.0
# Marius Hamacher: I simply made it compatible with tsv files...
# Usage: data_prep.pl /export/data/cv_corpus_v1/cv-valid-train valid_train

if (@ARGV != 3) {
  print STDERR "Usage: $0 <path-to-commonvoice-corpus> <dataset> <valid-train|valid-dev|valid-test>\n";
  print STDERR "e.g. $0 /export/data/cv_corpus_v1 cv-valid-train valid-train\n";
  exit(1);
}

($db_base, $dataset, $out_dir) = @ARGV;
$ext = "tsv";

open(CSV, "<", "$db_base/$dataset.$ext");

open(CSV, "<", "$db_base/$dataset.$ext") or die "cannot open dataset CSV file";
open(SPKR,">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(GNDR,">", "$out_dir/utt2gender") or die "Could not open the output file $out_dir/utt2gender";
open(TEXT,">", "$out_dir/text") or die "Could not open the output file $out_dir/text";
open(WAV,">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";
my $header = <CSV>;
while(<CSV>) {
  chomp;
  ($spkr, $filepath, $text, $upvotes, $downvotes, $age, $gender, $accent, $id, $synthetic, $duration) = split("\t", $_);
  if ("$gender" eq "female") {
    $gender = "f";
  } else {
    # Use male as default if not provided (no reason, just adopting the same default as in voxforge)
    $gender = "m";
  }
  $uttId = $filepath;
  if (-z "$db_base/clips/$filepath") {
    print "null file $filepath\n";
    next;
  }
  # speaker information should be suffix of the utterance Id
  $uttId = $id;
  $text =~ tr/a-z/A-Z/;
  if (index($text, "{") != -1 and index($text, "}" != -1)) {
    next;
  }
  print TEXT "$uttId"," ","$text","\n";
  print GNDR "$uttId"," ","$gender","\n";
  print WAV "$uttId"," ffmpeg -i \"$db_base/clips/$filepath\" -f wav -ar 16000 -ab 16 - |\n";
  print SPKR "$uttId"," $spkr","\n";
}
close(SPKR) || die;
close(TEXT) || die;
close(WAV) || die;
close(GNDR) || die;
close(WAVLIST);

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
system("env LC_COLLATE=C utils/fix_data_dir.sh $out_dir");
if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}