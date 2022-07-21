# import the countrycode library
# (see homepage for citation!: https://github.com/vincentarelbundock/countrycode)
library(countrycode)


###############################################################################
# TODO:



###############################################################################



# set main params
output_format = 'iso3c'
max_agrep_dist = 0.3


# manual fixes for countries that don't seem to be getting caught by countrycode
manuals_in = c('Sudan (former)')
manuals_out = c('South Sudan')
manuals = data.frame(input=manuals_in, output=manuals_out)


# quickie function to cat multiple blank lines
cat_blanks = function(n){
    cat(rep('\n', n))
}


# function for quick, assisted matching of countries
recode_country_vector = function(input,
                                 output_format='iso3c',
                                 allow_dups=TRUE,
                                 max_agrep_dist=0.3){

    # do manual replacement for any country names known to cause
    # problems that are not being automatically caught by countrycode
    for (row_n in range(nrow(manuals))){
        res = agrep(manuals[row_n,1], input, max.distance=max_agrep_dist)
        if (length(res) > 0){
            input[res] = manuals[row_n, 2]
        }
    }

    # get an initial vector of the attempted matches
    matches = countryname(input, destination=output_format)


    # create a lookup table to record correct codes for unmatched ones
    if (sum(is.na(matches)) > 0){
        # cat header before starting manual recoding
        cat_blanks(3)
        cat("Some strings failed to match!")
        cat_blanks(3)
        cat("About to begin assisted manual recoding. You will be asked to enter ISO-3 character codes.")
        cat_blanks(1)
        cat("These should be easy enough to find online, but here is one suggested helpful link:")
        cat_blanks(1)
        cat("\thttps://laendercode.net/en/countries.html")
        cat_blanks(3)
        lookup = data.frame(string=unique(input[is.na(matches)]), code=NA)
        cat('\nTHE FOLLOWING UNMATCHED STRINGS WILL NEED TO BE MANUALLY RECODED:\n\n')
        print(lookup)
        dummy = readline('\n\n\nPRESS <ENTER> TO BEGIN ASSISTED RECODE\n')
    } else {
        cat('\nNo strings failed to match!\n\n')
    }

    # check any NAs
    if (NA %in% matches){
        cat(rep('#', 80), sep='')
        # loop over and correct NAs
        na_idxs = which(is.na(matches))
        for (idx in na_idxs){
            # get that index's value
            not_matched = input[idx]
            if (is.na(lookup[lookup$string == not_matched,'code'])){
                cat_blanks(2)
                cat(paste0('The following value failed to match: ', not_matched))
                # use while loop to find the right match
                fixed = F
                while (!fixed){
                    # query user about it
                    cat_blanks(1)
                    new_val = readline('Please enter the ISO3C codes (3-letter code) for this country: ')
                    # attempt to match the new val
                    new_match = countrycode(new_val, origin='iso3c', destination=output_format, warn=F)
                    # check if that match worked
                    if (NA %in% new_match){
                        cat_blanks(1)
                        cat("Sorry. That didn't work. Please try again.")
                    } else {
                        # if it worked, store it and flip the fixed switch
                        cat_blanks(1)
                        cat("Excellent. Moving on...")
                        cat_blanks(2)
                        fixed = T
                        matches[idx] = new_match
                        lookup[lookup$string == not_matched,'code'] = new_match
                    }
                }
            }else{
                matches[idx] = lookup[lookup$string == not_matched,'code']
            }
        }
        cat(rep('-', 80), sep='')
        cat_blanks(3)
    }
    # check any duplicates, if required
    if (!allow_dups){
        cat(rep('#', 80), sep='')
        if (length(unique(matches)) < length(matches)){
           # get duplicate values
           dups = matches[duplicated(matches)] 
            # loop over and fix the dups
            for (dup in dups){
                cat_blanks(2)
                cat(paste0('The following value is duplicated in the output: ', dup))
                cat_blanks(1)
                cat("Here is a printout of all input values that returned this output:")
                cat_blanks(1)
                dup_idxs = which(matches == dup)
                print(input[dup_idxs])
                cat_blanks(1)
                fixed = F
                while (!fixed){
                    # ask for space or comma-separated ISO3C codes for each
                    new_vals = readline("Please enter the ISO3C (3-letter codes) for each of those inputs, separated by a comma or space: ")
                    cat_blanks(1)
                    # split the new input values, then try to match them
                    split_vals = strsplit(new_vals, "[ ,]")[[1]]
                    # match the new vals
                    new_matches = countrycode(split_vals, origin='iso3c',
                                              destination=output_format, warn=F)
                    # check if that match worked
                    if (NA %in% new_matches && length(duplicated(new_matches)) == 0) {
                        cat("Sorry. That didn't work. Please try again.")
                        cat_blanks(1)
                    } else {
                        # if it worked, store it and flip the fixed switch
                        cat("Excellent. Moving on...")
                        cat_blanks(1)
                        fixed = T
                        matches[dup_idxs] = new_matches
                    }
                }
            }
        cat(rep('-', 80), sep='')
        cat_blanks(3)

        }
    }

   # assert that there are no NAs and no duplicates left
   stopifnot(sum(is.na(matches)) == 0)
   if (!allow_dups){
        stopifnot(sum(duplicated(matches)) == 0)
   }
   
   # return the final vector of matches
   cat('')
   cat('All done! Be sure to double-check your results!\n\n')
   cat('')
   return(matches)
}
